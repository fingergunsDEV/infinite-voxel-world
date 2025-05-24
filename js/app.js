import * as THREE from "three";
import { PointerLockControls } from "PointerLockControls";
import { mergeBufferGeometries, mergeVertices } from "BufferGeometryUtils";

// --- Frustum Culling Setup ---
const frustum = new THREE.Frustum();
const projScreenMatrix = new THREE.Matrix4();

// --- Perlin Noise Implementation ---
/** Simplex noise by Stefan Gustavson, JavaScript version by Jonas Wagner (MIT license).
 *  Source: https://github.com/jwagner/simplex-noise.js (adapted below for browser direct use)
 */
class SimplexNoise {
	constructor(seed) {
		this.grad3 = new Float32Array([
			1,
			1,
			0,
			-1,
			1,
			0,
			1,
			-1,
			0,
			-1,
			-1,
			0,
			1,
			0,
			1,
			-1,
			0,
			1,
			1,
			0,
			-1,
			-1,
			0,
			-1,
			0,
			1,
			1,
			0,
			-1,
			1,
			0,
			1,
			-1,
			0,
			-1,
			-1
		]);
		this.p = this.buildPermutationTable(seed);
		this.perm = new Uint8Array(512);
		this.permMod12 = new Uint8Array(512);
		for (let i = 0; i < 512; i++) {
			this.perm[i] = this.p[i & 255];
			this.permMod12[i] = this.perm[i] % 12;
		}
	}
	buildPermutationTable(seed) {
		let p = new Uint8Array(256);
		for (let i = 0; i < 256; i++) p[i] = i;
		// Fisher–Yates
		let random = (() => {
			let s = seed || 1337;
			return () => (s = Math.imul(16807, s) % 2147483647) / 2147483647;
		})();
		for (let i = 255; i > 0; i--) {
			const r = Math.floor(random() * (i + 1));
			[p[i], p[r]] = [p[r], p[i]];
		}
		return p;
	}
	noise2D(xin, yin) {
		let permMod12 = this.permMod12,
			perm = this.perm,
			grad3 = this.grad3;
		let n0 = 0,
			n1 = 0,
			n2 = 0;
		let F2 = 0.5 * (Math.sqrt(3.0) - 1.0);
		let s = (xin + yin) * F2;
		let i = Math.floor(xin + s);
		let j = Math.floor(yin + s);
		let G2 = (3.0 - Math.sqrt(3.0)) / 6.0;
		let t = (i + j) * G2;
		let X0 = i - t;
		let Y0 = j - t;
		let x0 = xin - X0;
		let y0 = yin - Y0;
		let i1, j1;
		if (x0 > y0) {
			i1 = 1;
			j1 = 0;
		} else {
			i1 = 0;
			j1 = 1;
		}
		let x1 = x0 - i1 + G2;
		let y1 = y0 - j1 + G2;
		let x2 = x0 - 1.0 + 2.0 * G2;
		let y2 = y0 - 1.0 + 2.0 * G2;
		let ii = i & 255;
		let jj = j & 255;
		let gi0 = permMod12[ii + perm[jj]] * 3;
		let gi1 = permMod12[ii + i1 + perm[jj + j1]] * 3;
		let gi2 = permMod12[ii + 1 + perm[jj + 1]] * 3;
		let t0 = 0.5 - x0 * x0 - y0 * y0;
		if (t0 >= 0) {
			t0 *= t0;
			n0 = t0 * t0 * (grad3[gi0] * x0 + grad3[gi0 + 1] * y0);
		}
		let t1 = 0.5 - x1 * x1 - y1 * y1;
		if (t1 >= 0) {
			t1 *= t1;
			n1 = t1 * t1 * (grad3[gi1] * x1 + grad3[gi1 + 1] * y1);
		}
		let t2 = 0.5 - x2 * x2 - y2 * y2;
		if (t2 >= 0) {
			t2 *= t2;
			n2 = t2 * t2 * (grad3[gi2] * x2 + grad3[gi2 + 1] * y2);
		}
		return 70.0 * (n0 + n1 + n2);
	}
}

// Instantiate one global SimplexNoise
const simplex = new SimplexNoise(42);

// --- Config ---
const CHUNK_SIZE = 16;
const CHUNK_HEIGHT = 128;
let visibleRadius = 3;
const MIN_RENDER_DIST = 2;
const MAX_RENDER_DIST = 16;
const blocks = new Map();
const chunks = new Map();
const blockMaterials = {
	grass: new THREE.MeshStandardMaterial({ color: 0x228b22, flatShading: true }),
	sand: new THREE.MeshStandardMaterial({ color: 0xffd700, flatShading: true }),
	snow: new THREE.MeshStandardMaterial({ color: 0xffffff, flatShading: true }),
	stone: new THREE.MeshStandardMaterial({ color: 0x696969, flatShading: true }),
	water: new THREE.MeshStandardMaterial({
		color: 0x4169e1,
		transparent: true,
		opacity: 0.7,
		flatShading: true
	})
};
const highlightMaterial = new THREE.MeshBasicMaterial({
	color: 0xffffff,
	wireframe: true
});

// --- Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb);
// --- Fog Setup: The color should match the sky/background for blending effect ---
let fogEnabled = true;
let fogMin = 36;
let fogMax = 80;
const FOG_MIN_LIMIT = 20;
const FOG_MAX_LIMIT = 256;
const DEFAULT_FOG_NEAR = 36;
const DEFAULT_FOG_FAR = 80;
const FOG_BG_COLOR = 0x87ceeb;

function updateFog(fogNear, fogFar, enabled) {
	fogEnabled = enabled;
	if (enabled) {
		scene.fog = new THREE.Fog(FOG_BG_COLOR, fogNear, fogFar);
		scene.background = new THREE.Color(FOG_BG_COLOR);
	} else {
		scene.fog = null;
		// Optionally clear background color: keep consistent
		scene.background = new THREE.Color(FOG_BG_COLOR);
	}
}
updateFog(fogMin, fogMax, fogEnabled);
const camera = new THREE.PerspectiveCamera(
	75,
	window.innerWidth / window.innerHeight,
	0.1,
	1000
);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const gl = renderer.getContext();
const ext = gl.getExtension("EXT_occlusion_query_boolean");
if (!ext) console.warn("Occlusion queries not supported");
const boundingBoxGeometry = new THREE.BoxGeometry(
	CHUNK_SIZE,
	CHUNK_HEIGHT,
	CHUNK_SIZE
);
const boundingBoxMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
function attachOcclusion(mesh) {
	const { chunkX, chunkZ } = mesh.userData;
	const box = new THREE.Mesh(boundingBoxGeometry, boundingBoxMaterial);
	box.position.set(
		chunkX * CHUNK_SIZE + CHUNK_SIZE / 2,
		CHUNK_HEIGHT / 2,
		chunkZ * CHUNK_SIZE + CHUNK_SIZE / 2
	);
	mesh.userData.boundingBoxMesh = box;
	mesh.userData.query = gl.createQuery();
	mesh.userData.occlusionResult = true;
}

// --- Pointer Lock First Person Controls ---
const controls = new PointerLockControls(camera, renderer.domElement);
const moveSpeed = 0.07;
const gravity = 0.01;
const jumpStrength = 0.27;
const keys = {
	forward: false,
	backward: false,
	left: false,
	right: false,
	jump: false
};
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
const playerHeight = 1.8;
let onGround = false;

// --- Lighting ---
scene.add(new THREE.AmbientLight(0xffffff, 0.3));
const sun = new THREE.DirectionalLight(0xffffff, 1);
sun.position.set(50, 100, 50);
scene.add(sun);

// --- UI Overlay ---
const instructionOverlay = document.createElement("div");
function getControlHelpList() {
	return `
    <div>
      <b>Controls:</b><br>
      <span class="key">Click</span> to start / <span class="key">WASD</span> to move / <span class="key">Space</span> to jump<br>
      <span class="key">Left click</span>: Remove block<br>
      <span class="key">Right click</span>: Place block<br>
      <span class="key">E</span>: Change selected item<br>
      <span class="key">Q</span>: Drop item<br>
      <span class="key">F</span>: Collect nearby items<br>
      <span class="key">P</span>: Pause and settings menu<br>
    </div>`;
}
instructionOverlay.innerHTML = getControlHelpList();
Object.assign(instructionOverlay.style, {
	position: "fixed",
	top: "50%",
	left: "50%",
	transform: "translate(-50%,-50%)",
	fontSize: "22px",
	color: "white",
	zIndex: "1000",
	background: "rgba(30, 38, 50, 0.87)",
	padding: "32px 48px",
	borderRadius: "13px",
	textAlign: "center",
	boxShadow: "0 6px 42px #000b"
});
document.body.appendChild(instructionOverlay);
// -- Style keys:
const style = document.createElement("style");
style.textContent = `
.key {
    display: inline-block;
    background: #222d;
    color: #d0ffb2;
    font-size: 1em;
    padding: 1px 7px;
    margin: 0 2px;
    border-radius: 5px;
    font-weight: bold;
    border: 1.5px solid #42c37055;
}`;
document.head.appendChild(style);

// --- Crosshair SVG overlay ---
const crosshair = document.createElement("div");
crosshair.className = "crosshair";
crosshair.innerHTML = `
<svg width="32" height="32" viewBox="0 0 32 32">
  <g stroke="white" stroke-width="2">
    <line x1="16" y1="8" x2="16" y2="13"/>
    <line x1="16" y1="19" x2="16" y2="24"/>
    <line x1="8" y1="16" x2="13" y2="16"/>
    <line x1="19" y1="16" x2="24" y2="16"/>
  </g>
</svg>`;
document.body.appendChild(crosshair);

// --- Pointer Lock events ---
renderer.domElement.addEventListener("click", () => controls.lock());
controls.addEventListener("lock", () => {
	instructionOverlay.style.display = "none";
});
controls.addEventListener("unlock", () => {
	instructionOverlay.style.display = "block";
});

// --- Keyboard Movement Events ---
document.addEventListener("keydown", (event) => {
	switch (event.code) {
		case "KeyW":
			keys.forward = true;
			break;
		case "KeyS":
			keys.backward = true;
			break;
		case "KeyA":
			keys.left = true;
			break;
		case "KeyD":
			keys.right = true;
			break;
		case "Space":
			keys.jump = true;
			break;
		case "KeyE":
			// Toggle inventory selection
			const inventory = getInventory();
			inventory.selectedSlot =
				(inventory.selectedSlot + 1) % inventory.slots.length;
			updateInventoryUI();
			break;
		case "KeyQ":
			// Drop item
			if (inventory.slots[inventory.selectedSlot].count > 0) {
				dropItem();
			}
			break;
		case "KeyF":
			// Interact/collect nearby items
			collectNearbyItems();
			break;
	}
});
document.addEventListener("keyup", (event) => {
	switch (event.code) {
		case "KeyW":
			keys.forward = false;
			break;
		case "KeyS":
			keys.backward = false;
			break;
		case "KeyA":
			keys.left = false;
			break;
		case "KeyD":
			keys.right = false;
			break;
		case "Space":
			keys.jump = false;
			break;
	}
});

// --- Terrain Height Function with Chunk Caching ---
// Cache of height map per chunk (size: CHUNK_SIZE × CHUNK_SIZE)
const heightCache = new Map();

/** Pure noise‐based height computation (original getHeight logic) */
function computeHeight(x, z) {
	// Layer 1: Base, mostly flat
	let base =
		simplex.noise2D(x * 0.006, z * 0.006) * 1.8 +
		simplex.noise2D((x + 500) * 0.013, (z - 888) * 0.013) * 0.4;
	base = base * 8 + 50;

	// Layer 2: Sparse Mountains
	let mountainNoise = simplex.noise2D(x * 0.0025 + 3000, z * 0.0025 - 735);
	if (mountainNoise > 0.52) {
		let m = Math.min((mountainNoise - 0.52) * 1.9, 1);
		let detail = simplex.noise2D(x * 0.022, z * 0.022) * 0.7;
		let cliff = Math.pow(m, 2.6);
		base += cliff * 38 + detail * 2;
	}

	// Layer 3: Rivers (deeper, more defined)
	let riverNoise = simplex.noise2D(x * 0.002, 9999 + z * 0.002);
	let riverBand = Math.abs(riverNoise);
	if (riverBand < 0.15) {
		let t = Math.pow(riverBand / 0.15, 1.7);
		let riverDepth = (1 - t) * 24 + simplex.noise2D(x * 0.032, z * 0.032) * 2.5;
		base -= riverDepth;
	}

	// Clamp and floor to [0, CHUNK_HEIGHT–1]
	return Math.floor(Math.max(0, Math.min(base, CHUNK_HEIGHT - 1)));
}

/** Cached wrapper: stores computed heights per‐chunk so each (x,z) only pays the noise cost once */
function getHeight(x, z) {
	const chunkX = Math.floor(x / CHUNK_SIZE);
	const chunkZ = Math.floor(z / CHUNK_SIZE);
	const cacheKey = `${chunkX},${chunkZ}`;

	let arr = heightCache.get(cacheKey);
	if (!arr) {
		arr = new Int16Array(CHUNK_SIZE * CHUNK_SIZE);
		arr.fill(-1);
		heightCache.set(cacheKey, arr);
	}

	const localX = x - chunkX * CHUNK_SIZE;
	const localZ = z - chunkZ * CHUNK_SIZE;
	const idx = localZ * CHUNK_SIZE + localX;
	let h = arr[idx];

	if (h < 0) {
		h = computeHeight(x, z);
		arr[idx] = h;
	}
	return h;
}

// --- Block Existence Query ---
function blockKey(x, y, z) {
	return `${x},${y},${z}`;
}
function hasBlock(x, y, z) {
	if (y < 0 || y >= CHUNK_HEIGHT) return false;
	const key = blockKey(x, y, z);
	if (blocks.has(key)) return blocks.get(key);
	return y <= getHeight(x, z);
}
// Add or remove block for world (true for present, false for absent)
function setBlock(x, y, z, present) {
	const key = blockKey(x, y, z);
	blocks.set(key, present);
	recordChunkEdit(x, y, z, present);

	// Maintain spatial grid
	if (present) {
		playerEditGrid.addBlock(x, y, z);
	} else {
		playerEditGrid.removeBlock(x, y, z);
	}

	// Only update/rebuild affected chunks (NOT remove, so no flicker!)
	for (const chunkKey of getTouchedChunks(x, y, z)) {
		const [chunkX, chunkZ] = chunkKey.split(",").map(Number);
		requestChunkRebuild(chunkX, chunkZ);
	}
}

// --- Geometry ---
function createCubeGeometry(x, y, z) {
	const geometry = new THREE.BoxGeometry(1, 1, 1);
	geometry.translate(x + 0.5, y + 0.5, z + 0.5);
	return geometry;
}

// Create a single face geometry at block (x,y,z) with given normal
function createFaceGeometry(x, y, z, normal) {
	const plane = new THREE.PlaneGeometry(1, 1);
	const quat = new THREE.Quaternion();
	quat.setFromUnitVectors(
		new THREE.Vector3(0, 0, 1),
		new THREE.Vector3(normal[0], normal[1], normal[2])
	);
	plane.applyQuaternion(quat);
	plane.translate(
		x + 0.5 + normal[0] * 0.5,
		y + 0.5 + normal[1] * 0.5,
		z + 0.5 + normal[2] * 0.5
	);
	return plane;
}

// Helper: test whether block (x, y, z) is considered "naturally-solid" in terrain (i.e., not a user-placed or removed block)
function isNaturalSolidBlock(x, y, z) {
	if (y < 0 || y >= CHUNK_HEIGHT) return false;
	const key = blockKey(x, y, z);
	if (blocks.has(key)) return false; // user modified – not natural
	return y <= getHeight(x, z);
}

// Returns merged geometry of chunk at (chunkX, chunkZ) with per-face culling & boundingSphere
function generateChunk(chunkX, chunkZ) {
	const geometries = [];
	const materials = new Map();
	const startX = chunkX * CHUNK_SIZE;
	const startZ = chunkZ * CHUNK_SIZE;

	// Spawn items when generating new chunks
	spawnItemsInChunk(chunkX, chunkZ);

	for (let x = 0; x < CHUNK_SIZE; x++) {
		for (let z = 0; z < CHUNK_SIZE; z++) {
			const worldX = startX + x;
			const worldZ = startZ + z;
			let height = CHUNK_HEIGHT - 1;
			// Find top-most block in the terrain (natural or user-added), so we don't iterate excessive heights
			while (height >= 0 && !hasBlock(worldX, height, worldZ)) height--;
			for (let y = 0; y <= height; y++) {
				if (!hasBlock(worldX, y, worldZ)) continue;

				// Modified Face-Culling:
				// - Only cull face (do NOT draw face) if BOTH this block and its neighbor
				//   are *unmodified, natural terrain blocks*.
				// - If EITHER is player-placed/removed, always render face if neighbor is not present.
				// This ensures that player-created caves and any exposed block inside a "dug" region gets visible faces.

				const faceDirs = [
					[0, 1, 0],
					[0, -1, 0],
					[1, 0, 0],
					[-1, 0, 0],
					[0, 0, 1],
					[0, 0, -1]
				];
				const currIsNatural = isNaturalSolidBlock(worldX, y, worldZ);

				for (const normal of faceDirs) {
					const nx = worldX + normal[0],
						ny = y + normal[1],
						nz = worldZ + normal[2];
					const neighborExists = hasBlock(nx, ny, nz);
					const neighborIsNatural = isNaturalSolidBlock(nx, ny, nz);
					// Draw face if neighbor is air OR either block was modified by player
					if (!neighborExists || !(currIsNatural && neighborIsNatural)) {
						const material = getBlockMaterial(worldX, y, worldZ);
						if (!materials.has(material)) {
							materials.set(material, []);
						}

						materials
							.get(material)
							.push(createFaceGeometry(worldX, y, worldZ, normal));
					}
				}
			}
		}
	}

	if (materials.size === 0) return null;

	// Create merged geometry for each material type
	const meshes = [];
	for (const [material, faceGeometries] of materials.entries()) {
		if (faceGeometries.length > 0) {
			let chunkGeo = mergeBufferGeometries(faceGeometries, false);
			chunkGeo = mergeVertices(chunkGeo);
			chunkGeo.computeBoundingSphere();
			const mesh = new THREE.Mesh(chunkGeo, material);
			meshes.push(mesh);
		}
	}

	// Create a group to hold all material meshes
	const chunkGroup = new THREE.Group();
	meshes.forEach((mesh) => chunkGroup.add(mesh));
	return chunkGroup;
}

// --- Chunk Handling ---
const activeRadius = 1.5; // Only activate meshes within this many chunks from the player (smaller than visible radius for perf)

function isChunkActive(chunkX, chunkZ, camChunkX, camChunkZ) {
	const dx = chunkX - camChunkX;
	const dz = chunkZ - camChunkZ;
	const dist = Math.sqrt(dx * dx + dz * dz);
	return dist <= activeRadius;
}

let globalChunkBuildID = 1;

/**
 * Schedule a chunk geometry update.
 * Instead of removing the existing mesh immediately, mark it as "pending", then swap new geometry in place.
 */
function requestChunkRebuild(chunkX, chunkZ) {
	const key = `${chunkX},${chunkZ}`;
	const chunk = chunks.get(key);
	// If chunk not loaded yet, do nothing; load cycle will build correct geometry.
	if (!chunk || chunk.userData.state === "pending") return;
	// Mark as pending, bump buildID
	chunk.userData.pendingBuildVersion = ++globalChunkBuildID;
	const requestedVersion = chunk.userData.pendingBuildVersion;
	// Request geometry (async), then swap mesh in
	requestChunkGeometry(chunkX, chunkZ, key).then((faces) => {
		// It is possible for another rebuild to have been scheduled since, so ensure only latest is used.
		if (!chunks.has(key)) return;
		const existing = chunks.get(key);
		if (
			!existing.userData.pendingBuildVersion ||
			existing.userData.pendingBuildVersion !== requestedVersion
		) {
			// Outdated build, ignore result.
			return;
		}
		// Create new mesh
		const newMesh = new THREE.Mesh(
			facesToBufferGeometry(faces),
			blockMaterials.grass
		);
		newMesh.frustumCulled = true;
		newMesh.userData = {
			chunkX,
			chunkZ,
			state: "ready",
			buildVersion: requestedVersion
		};
		newMesh.visible = existing.visible;
		attachOcclusion(newMesh);
		// Swap mesh in the scene, preserving chunk Map key
		scene.add(newMesh);
		scene.remove(existing);
		// Clean up
		if (existing.geometry) existing.geometry.dispose?.();
		if (existing.material) existing.material.dispose?.();
		chunks.set(key, newMesh);
	});
}

let chunkLoadQueue = [];
let isChunkLoading = false;

// Modified addChunk to return a promise
function addChunk(chunkX, chunkZ) {
	const key = `${chunkX},${chunkZ}`;
	if (chunks.has(key)) return Promise.resolve();
	const dummy = new THREE.Object3D();
	dummy.visible = false;
	dummy.userData = {
		chunkX,
		chunkZ,
		state: "pending",
		buildVersion: ++globalChunkBuildID
	};
	scene.add(dummy);
	chunks.set(key, dummy);
	const thisBuildID = dummy.userData.buildVersion;
	const promise = requestChunkGeometry(chunkX, chunkZ, key).then((faces) => {
		if (!chunks.has(key)) return;
		const existing = chunks.get(key);
		if (
			existing.userData.buildVersion !== thisBuildID &&
			existing.userData.pendingBuildVersion !== thisBuildID
		) {
			return;
		}
		const mesh = new THREE.Mesh(
			facesToBufferGeometry(faces),
			blockMaterials.grass
		);
		mesh.frustumCulled = true;
		mesh.userData = {
			chunkX,
			chunkZ,
			state: "ready",
			buildVersion: thisBuildID
		};
		mesh.visible = existing.visible;
		attachOcclusion(mesh);
		scene.add(mesh);
		scene.remove(existing);
		if (existing.geometry) existing.geometry.dispose?.();
		if (existing.material) existing.material.dispose?.();
		chunks.set(key, mesh);
	});
	return promise;
}

// Process the next chunk in the load queue
function processChunkLoadQueue() {
	if (isChunkLoading || chunkLoadQueue.length === 0) return;
	const [chunkX, chunkZ] = chunkLoadQueue.shift();
	isChunkLoading = true;
	addChunk(chunkX, chunkZ).finally(() => {
		isChunkLoading = false;
		processChunkLoadQueue();
	});
}

// --- Player collision helper ---
function getGroundHeight(x, z) {
	for (let y = CHUNK_HEIGHT - 1; y >= 0; y--)
		if (hasBlock(Math.floor(x), y, Math.floor(z))) return y + 1;
	return 0;
}

// Test if the player is on ground (distance at feet to ground <0.1)
function isPlayerOnGround(p) {
	const yFeet = p.y - playerHeight;
	const groundY = getGroundHeight(p.x, p.z);
	return Math.abs(yFeet - groundY) < 0.08;
}

// More accurate (step-)collision check
function tryHorizontalMove(newPos) {
	const rad = 0.3;
	const minX = newPos.x - rad,
		maxX = newPos.x + rad;
	const minY = newPos.y - playerHeight,
		maxY = newPos.y;
	const minZ = newPos.z - rad,
		maxZ = newPos.z + rad;
	const x0 = Math.floor(minX),
		x1 = Math.floor(maxX);
	const y0 = Math.floor(minY),
		y1 = Math.floor(maxY);
	const z0 = Math.floor(minZ),
		z1 = Math.floor(maxZ);
	for (let x = x0; x <= x1; x++) {
		for (let y = y0; y <= y1; y++) {
			for (let z = z0; z <= z1; z++) {
				if (hasBlock(x, y, z)) {
					if (
						minX < x + 1 &&
						maxX > x &&
						minY < y + 1 &&
						maxY > y &&
						minZ < z + 1 &&
						maxZ > z
					) {
						return false;
					}
				}
			}
		}
	}
	return true;
}

// AABB collision detection for player
function checkCollision(pos) {
	const radius = 0.3;
	const minX = pos.x - radius,
		maxX = pos.x + radius;
	const minY = pos.y - playerHeight,
		maxY = pos.y;
	const minZ = pos.z - radius,
		maxZ = pos.z + radius;
	const footY = minY;
	const epsilon = 0.001;

	// 1. User-modified blocks via grid:
	const gridBlocks = playerEditGrid.queryAABB(
		minX,
		maxX,
		minY,
		maxY,
		minZ,
		maxZ
	);
	for (const [x, y, z] of gridBlocks) {
		// Only collide with present blocks
		if (!hasBlock(x, y, z)) continue;
		const blockMinX = x,
			blockMaxX = x + 1;
		const blockMinY = y,
			blockMaxY = y + 1;
		const blockMinZ = z,
			blockMaxZ = z + 1;
		if (blockMaxY <= footY + epsilon) continue;
		if (
			minX < blockMaxX &&
			maxX > blockMinX &&
			minY < blockMaxY &&
			maxY > blockMinY &&
			minZ < blockMaxZ &&
			maxZ > blockMinZ
		) {
			return true;
		}
	}

	// 2. Procedural: If no collision with user blocks, check procedural blocks only
	const x0 = Math.floor(minX),
		x1 = Math.floor(maxX);
	const y0 = Math.floor(minY),
		y1 = Math.floor(maxY);
	const z0 = Math.floor(minZ),
		z1 = Math.floor(maxZ);
	for (let x = x0; x <= x1; x++) {
		for (let y = y0; y <= y1; y++) {
			for (let z = z0; z <= z1; z++) {
				const key = blockKey(x, y, z);
				// Only include *non-user-modified* blocks from procedural terrain
				if (!blocks.has(key) && y <= getHeight(x, z)) {
					const blockMinX = x,
						blockMaxX = x + 1;
					const blockMinY = y,
						blockMaxY = y + 1;
					const blockMinZ = z,
						blockMaxZ = z + 1;
					if (blockMaxY <= footY + epsilon) continue;
					if (
						minX < blockMaxX &&
						maxX > blockMinX &&
						minY < blockMaxY &&
						maxY > blockMinY &&
						minZ < blockMaxZ &&
						maxZ > blockMinZ
					) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

// --- Raycasting for block interactions ---
const raycaster = new THREE.Raycaster();
const mouse = { x: 0, y: 0 };
const highlightMesh = new THREE.Mesh(
	new THREE.BoxGeometry(1.01, 1.01, 1.01),
	highlightMaterial
);
highlightMesh.visible = false;
scene.add(highlightMesh);
let highlightBlock = null,
	highlightNormal = null;

// Improved: Returns the targeted block (the block being looked at, and the normal of the face hit)
function getTargetedBlock() {
	// Raycast from camera center
	raycaster.setFromCamera({ x: 0, y: 0 }, camera);

	const maxDist = 6;
	const step = 0.04; // increased raymarch precision
	let lastAir = null;
	let lastBlock = null;
	let lastNormal = null;

	// Use a "raymarching" approach for robust block targeting
	for (let t = 0; t <= maxDist; t += step) {
		const pos = camera.position
			.clone()
			.add(camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(t));
		const bx = Math.floor(pos.x),
			by = Math.floor(pos.y),
			bz = Math.floor(pos.z);

		if (hasBlock(bx, by, bz)) {
			// Compute face normal by looking at what direction the ray entered this block from previous "air"
			if (lastAir) {
				// Determine normal as the largest delta between lastAir and this block
				const dx = bx - lastAir[0];
				const dy = by - lastAir[1];
				const dz = bz - lastAir[2];
				let normal = [0, 0, 0];
				if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > Math.abs(dz))
					normal = [dx, 0, 0];
				else if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > Math.abs(dz))
					normal = [0, dy, 0];
				else normal = [0, 0, dz];
				return {
					pos: [bx, by, bz],
					normal,
					dist: t,
					placePos: [lastAir[0], lastAir[1], lastAir[2]]
				};
			} else {
				// If ray started inside a block, just use this block, and arbitrary normal
				return {
					pos: [bx, by, bz],
					normal: [0, 1, 0],
					dist: t,
					placePos: [bx, by + 1, bz]
				};
			}
		} else {
			lastAir = [bx, by, bz];
		}
	}
	// No block targeted
	return null;
}

// --- Utility: get all chunks affecting/affected by placement/removal at (x, y, z) ---
function getTouchedChunks(x, y, z) {
	// Placing/removing a block can affect the geometry in up to 6 neighbor chunks (sharing a face)
	// Identify the chunk for (x,y,z) plus any neighbor chunk in directions where the block is on the edge
	const touched = new Set();
	const cx = Math.floor(x / CHUNK_SIZE);
	const cz = Math.floor(z / CHUNK_SIZE);
	touched.add(`${cx},${cz}`);

	// Check if the block is at the +/- X or Z edge of a chunk - if so, add neighbor chunk sharing that face
	if (x % CHUNK_SIZE === 0) {
		touched.add(`${cx - 1},${cz}`);
	}
	if ((x + 1) % CHUNK_SIZE === 0) {
		touched.add(`${cx + 1},${cz}`);
	}
	if (z % CHUNK_SIZE === 0) {
		touched.add(`${cx},${cz - 1}`);
	}
	if ((z + 1) % CHUNK_SIZE === 0) {
		touched.add(`${cx},${cz + 1}`);
	}
	// (Diagonal chunks are not affected unless corners matter)
	return touched;
}

// --- Spatial Partitioning for Collision Checks: Uniform Grid ---
/**
 * A spatial hash/grid to accelerate collision queries for user-modified blocks.
 * Voxel terrain from the procedural generator (unchanged blocks) is not stored here;
 * only player-placed or removed blocks are tracked. On collision, first probe the grid,
 * then fallback to procedural check as needed.
 */
// 1. Partition world space into 8x8x8 block regions
const GRID_SIZE = 8; // size of each grid cell in blocks

class SpatialGrid {
	constructor(cellSize = GRID_SIZE) {
		this.cellSize = cellSize;
		this.cells = new Map(); // Map key: `${gx},${gy},${gz}` -> Set of [x,y,z] block arrays
	}
	// Hash grid coords
	_keyFromXYZ(x, y, z) {
		const gx = Math.floor(x / this.cellSize);
		const gy = Math.floor(y / this.cellSize);
		const gz = Math.floor(z / this.cellSize);
		return `${gx},${gy},${gz}`;
	}
	// Get affected grid cells (may result in multiple for objects spanning multiple cells)
	_getCellCoords(x, y, z) {
		return [
			Math.floor(x / this.cellSize),
			Math.floor(y / this.cellSize),
			Math.floor(z / this.cellSize)
		];
	}
	// Insert block at integer position
	addBlock(x, y, z) {
		const key = this._keyFromXYZ(x, y, z);
		if (!this.cells.has(key)) this.cells.set(key, new Set());
		this.cells.get(key).add(`${x},${y},${z}`);
	}
	// Remove block at integer position
	removeBlock(x, y, z) {
		const key = this._keyFromXYZ(x, y, z);
		const cell = this.cells.get(key);
		if (cell) cell.delete(`${x},${y},${z}`);
	}
	// Query blocks within an AABB: [minX, maxX, minY, maxY, minZ, maxZ]
	queryAABB(minX, maxX, minY, maxY, minZ, maxZ) {
		const results = [];
		// Compute grid cell bounds to check
		const gx0 = Math.floor(minX / this.cellSize);
		const gx1 = Math.floor(maxX / this.cellSize);
		const gy0 = Math.floor(minY / this.cellSize);
		const gy1 = Math.floor(maxY / this.cellSize);
		const gz0 = Math.floor(minZ / this.cellSize);
		const gz1 = Math.floor(maxZ / this.cellSize);

		for (let gx = gx0; gx <= gx1; gx++) {
			for (let gy = gy0; gy <= gy1; gy++) {
				for (let gz = gz0; gz <= gz1; gz++) {
					const key = `${gx},${gy},${gz}`;
					const cell = this.cells.get(key);
					if (cell && cell.size) {
						for (const idx of cell) {
							const [bx, by, bz] = idx.split(",").map(Number);
							if (
								bx + 1 > minX &&
								bx < maxX + 1 &&
								by + 1 > minY &&
								by < maxY + 1 &&
								bz + 1 > minZ &&
								bz < maxZ + 1
							) {
								results.push([bx, by, bz]);
							}
						}
					}
				}
			}
		}
		return results;
	}
	// When reloading/clearing edits for a chunk, we must remove those blocks from the grid.
	flushChunkEdits(edits) {
		// Edits is an object of { blockKey(x,y,z): present }
		for (const k in edits) {
			const [x, y, z] = k.split(",").map(Number);
			this.removeBlock(x, y, z);
		}
	}
	// When loading chunk edits, re-add present ones
	addChunkEdits(edits) {
		for (const k in edits) {
			const [x, y, z] = k.split(",").map(Number);
			if (edits[k]) this.addBlock(x, y, z);
		}
	}
}

// Instantiate a single grid for all player-modified blocks
const playerEditGrid = new SpatialGrid(GRID_SIZE);

// On load, populate grid with any existing user-modified blocks (should be empty initially)
(function () {
	for (const [k, v] of blocks.entries()) {
		if (v) {
			const [x, y, z] = k.split(",").map(Number);
			playerEditGrid.addBlock(x, y, z);
		}
	}
})();

// --- PAUSE MENU SETUP ---
const pauseMenu = document.createElement("div");
pauseMenu.id = "pauseMenu";
pauseMenu.tabIndex = -1;
pauseMenu.innerHTML = `
  <div class="pause-content">
    <h2>Paused</h2>
    <label for="renderDistanceSlider">Render Distance:
      <span id="renderDistanceValue"></span>
    </label>
    <input type="range" id="renderDistanceSlider" min="${MIN_RENDER_DIST}" max="${MAX_RENDER_DIST}" step="1">
    <label for="fogDistanceSlider" style="margin-top:19px;">
      Fog Distance:
      <span id="fogDistanceValue"></span>
    </label>
    <input type="range" id="fogDistanceSlider" min="${FOG_MIN_LIMIT}" max="${FOG_MAX_LIMIT}" step="1" value="${DEFAULT_FOG_FAR}">
    <label style="margin-top: 9px; display: flex; align-items: center; gap: 10px; font-size: 1em;">
      <input type="checkbox" id="fogEnableCheckbox" checked style="margin:0;">
      <span style="user-select:none;">Enable Fog</span>
    </label>
    <button id="resumeBtn">Resume</button>
    <p style="margin-top:18px;font-size:.94em;color:#9adca7;font-style:italic">Press <b>P</b> to resume or use the button</p>
  </div>
`;
document.body.appendChild(pauseMenu);

const renderDistanceSlider = pauseMenu.querySelector("#renderDistanceSlider");
const renderDistanceValue = pauseMenu.querySelector("#renderDistanceValue");
const resumeBtn = pauseMenu.querySelector("#resumeBtn");
const pauseContent = pauseMenu.querySelector(".pause-content");

// FOG controls
const fogDistanceSlider = pauseMenu.querySelector("#fogDistanceSlider");
const fogDistanceValue = pauseMenu.querySelector("#fogDistanceValue");
const fogEnableCheckbox = pauseMenu.querySelector("#fogEnableCheckbox");

// Init values
renderDistanceSlider.value = visibleRadius;
renderDistanceValue.textContent = visibleRadius;
fogDistanceSlider.value = fogMax;
fogDistanceValue.textContent = fogMax + (fogEnabled ? " units" : " (disabled)");
fogEnableCheckbox.checked = fogEnabled;

// PAUSE/RESUME LOGIC
let isPaused = false;
function setPaused(paused) {
	if (paused === isPaused) return;
	isPaused = paused;
	if (paused) {
		document.body.classList.add("pause-active");
		pauseMenu.style.display = "flex";
		// Unlock pointer lock if active
		controls.unlock();
	} else {
		document.body.classList.remove("pause-active");
		pauseMenu.style.display = "none";
		// Try to regain pointer lock if not on overlay
		// User must click to regain
	}
}

// Listen for P key to pause/resume
document.addEventListener("keydown", (e) => {
	if (e.code === "KeyP" && !e.repeat) {
		setPaused(!isPaused);
	}
});
resumeBtn.addEventListener("click", () => setPaused(false));
pauseMenu.addEventListener("keydown", (e) => {
	if (e.code === "KeyP" && isPaused) {
		setPaused(false);
	}
});

// When menu is open, prevent tab from leaving menu, also stop propagation to avoid accidental controls
pauseMenu.addEventListener("keydown", (e) => {
	if (e.code === "Tab") {
		e.preventDefault();
		pauseContent.focus();
	}
	e.stopPropagation();
});

function setFogSliderAndValue(display, enableState) {
	fogDistanceValue.textContent =
		display + (enableState ? " units" : " (disabled)");
}

// On slider change (fog distance)
fogDistanceSlider.addEventListener("input", (e) => {
	fogMax = Number(e.target.value);
	if (fogEnabled) updateFog(fogMin, fogMax, true);
	setFogSliderAndValue(fogMax, fogEnabled);
});
// Checkbox: Enable/disable fog
fogEnableCheckbox.addEventListener("input", (e) => {
	fogEnabled = !!e.target.checked;
	if (fogEnabled) {
		updateFog(fogMin, fogMax, true);
	} else {
		updateFog(fogMin, fogMax, false);
	}
	setFogSliderAndValue(fogDistanceSlider.value, fogEnabled);
});

// Utility: handle update when changing slider from code
function syncFogStateToUI() {
	fogEnableCheckbox.checked = fogEnabled;
	fogDistanceSlider.value = fogMax;
	setFogSliderAndValue(fogMax, fogEnabled);
}

// On render distance change
function setVisibleRadius(val) {
	visibleRadius = Math.max(
		MIN_RENDER_DIST,
		Math.min(MAX_RENDER_DIST, Number(val))
	);
	renderDistanceValue.textContent = visibleRadius;
	updateChunks();
}
renderDistanceSlider.addEventListener("input", (e) =>
	setVisibleRadius(e.target.value)
);

// Hide pause menu on start
pauseMenu.style.display = "none";

// --- Chunk Add/Remove and Visibility Management ---
function updateChunks() {
	// Determine current camera chunk
	const camChunkX = Math.floor(camera.position.x / CHUNK_SIZE);
	const camChunkZ = Math.floor(camera.position.z / CHUNK_SIZE);
	const radius = visibleRadius;

	// Gather desired chunks within radius
	const desired = [];
	for (let dx = -radius; dx <= radius; dx++) {
		for (let dz = -radius; dz <= radius; dz++) {
			if (Math.hypot(dx, dz) <= radius) {
				desired.push([camChunkX + dx, camChunkZ + dz]);
			}
		}
	}
	const desiredKeys = new Set(desired.map(([x, z]) => `${x},${z}`));

	// Unload chunks that are no longer within radius
	for (const [key, mesh] of chunks.entries()) {
		if (!desiredKeys.has(key)) {
			scene.remove(mesh);
			if (mesh.geometry) mesh.geometry.dispose();
			if (mesh.material) mesh.material.dispose();
			chunks.delete(key);
		}
	}

	// Sort load queue by proximity to camera
	chunkLoadQueue = desired.sort((a, b) => {
		const da = Math.hypot(a[0] - camChunkX, a[1] - camChunkZ);
		const db = Math.hypot(b[0] - camChunkX, b[1] - camChunkZ);
		return da - db;
	});

	// Start processing the queue
	processChunkLoadQueue();
}

// --- Worker Pool & Parallel Chunk Meshing ---
const WORKER_POOL_SIZE = Math.max(
	2,
	Math.min(6, navigator.hardwareConcurrency || 4)
);
const pendingChunkJobs = new Map();
const chunkPool = [];
const workerPool = [];
let chunkWorkerBlobURL = null;

// Track max concurrency and timing for CPU load
let maxConcurrentWorkers = Math.max(
	2,
	Math.min(WORKER_POOL_SIZE, navigator.hardwareConcurrency || 4)
);
let chunkJobQueue = []; // {chunkX, chunkZ, resolve, reject, editObj}
let outstandingJobs = 0;
let recentJobTimes = []; // [ms,ms,...] for dynamic load management

function setMaxConcurrentWorkers(count) {
	maxConcurrentWorkers = Math.min(WORKER_POOL_SIZE, Math.max(1, count));
}
function considerThrottlingJobs(avgTime) {
	// If jobs are slow, reduce concurrency; if fast, up to allowed maximum
	if (avgTime > 70 && maxConcurrentWorkers > 2) {
		setMaxConcurrentWorkers(maxConcurrentWorkers - 1);
	} else if (avgTime < 30 && maxConcurrentWorkers < WORKER_POOL_SIZE) {
		setMaxConcurrentWorkers(maxConcurrentWorkers + 1);
	}
}
function updateJobTimings(time) {
	recentJobTimes.push(time);
	if (recentJobTimes.length > 23) recentJobTimes.shift();
	if (recentJobTimes.length === 23) {
		const avg = recentJobTimes.reduce((a, b) => a + b, 0) / recentJobTimes.length;
		considerThrottlingJobs(avg);
	}
}

// Create a worker "chunk mesher" from an inline Blob
function getChunkWorkerBlob() {
	return new Blob(
		[
			`
    // Worker: chunk geometry builder
    self.importScripts();
    class SimplexNoise {
        constructor(seed) {
            this.grad3 = new Float32Array([
                1,1,0,-1,1,0,1,-1,0,-1,-1,0,
                1,0,1,-1,0,1,1,0,-1,-1,0,-1,
                0,1,1,0,-1,1,0,1,-1,0,-1,-1]);
            this.p = this.buildPermutationTable(seed);
            this.perm = new Uint8Array(512);
            this.permMod12 = new Uint8Array(512);
            for (let i=0;i<512;i++) {
                this.perm[i]=this.p[i&255];
                this.permMod12[i] = this.perm[i] % 12;
            }
        }
        buildPermutationTable(seed) {
            let p = new Uint8Array(256);
            for (let i=0;i<256;i++) p[i]=i;
            let random = (() => {
                let s = (seed||1337);
                return () => (s=Math.imul(16807,s)%2147483647)/2147483647;
            })();
            for (let i=255;i>0;i--) {
                const r = Math.floor(random()*(i+1));
                [p[i],p[r]] = [p[r],p[i]];
            }
            return p;
        }
        noise2D(xin, yin) {
          let permMod12 = this.permMod12, perm = this.perm, grad3 = this.grad3;
          let n0 = 0, n1 = 0, n2 = 0;
          let F2 = 0.5 * (Math.sqrt(3.0) - 1.0);
          let s = (xin + yin) * F2;
          let i = Math.floor(xin + s);
          let j = Math.floor(yin + s);
          let G2 = (3.0 - Math.sqrt(3.0)) / 6.0;
          let t = (i + j) * G2;
          let X0 = i - t;
          let Y0 = j - t;
          let x0 = xin - X0;
          let y0 = yin - Y0;
          let i1, j1;
          if(x0 > y0) {i1=1; j1=0;} else {i1=0;j1=1;}
          let x1 = x0 - i1 + G2;
          let y1 = y0 - j1 + G2;
          let x2 = x0 - 1.0 + 2.0 * G2;
          let y2 = y0 - 1.0 + 2.0 * G2;
          let ii = i & 255;
          let jj = j & 255;
          let gi0 = permMod12[ii + perm[jj]] * 3;
          let gi1 = permMod12[ii + i1 + perm[jj + j1]] * 3;
          let gi2 = permMod12[ii + 1 + perm[jj + 1]] * 3;
          let t0 = 0.5 - x0 * x0 - y0 * y0;
          if(t0 >= 0) {
            t0 *= t0;
            n0 = t0 * t0 * (grad3[gi0] * x0 + grad3[gi0 + 1] * y0);
          }
          let t1 = 0.5 - x1 * x1 - y1 * y1;
          if(t1 >= 0) {
            t1 *= t1;
            n1 = t1 * t1 * (grad3[gi1] * x1 + grad3[gi1 + 1] * y1);
          }
          let t2 = 0.5 - x2 * x2 - y2 * y2;
          if(t2 >= 0) {
            t2 *= t2;
            n2 = t2 * t2 * (grad3[gi2] * x2 + grad3[gi2 + 1] * y2);
          }
          return 70.0 * (n0 + n1 + n2);
        }
    }
    let simplex = new SimplexNoise(42);
    let CHUNK_SIZE, CHUNK_HEIGHT;

    function getHeight(x, z) {
        let base = simplex.noise2D(x * 0.006, z * 0.006) * 1.8
            + simplex.noise2D((x + 500) * 0.013, (z - 888) * 0.013) * 0.4;
        base = base * 8 + 50;
        let mountainNoise = simplex.noise2D(x * 0.0025 + 3000, z * 0.0025 - 735);
        let mountainHeight = 0;
        if (mountainNoise > 0.52) {
            let m = (mountainNoise - 0.52) * 1.9;
            m = Math.min(m, 1);
            let detail = simplex.noise2D(x * 0.022, z * 0.022) * 0.7;
            let cliff = Math.pow(m, 2.6);
            mountainHeight = cliff * 38 + detail * 2;
            base += mountainHeight;
        }
        let riverNoise = simplex.noise2D(x * 0.002, 9999 + z * 0.002);
        let riverBand = Math.abs(riverNoise);
        if (riverBand < 0.15) {
            let t = (riverBand / 0.15); 
            t = Math.pow(t, 1.7); 
            let riverDepth = (1 - t) * 24 + simplex.noise2D(x * 0.032, z * 0.032) * 2.5;
            base -= riverDepth;
        }
        return Math.floor(Math.max(0, Math.min(base, CHUNK_HEIGHT - 1)));
    }
    function blockKey(x,y,z) { return x+','+y+','+z; }
    function hasBlock(x, y, z, editedBlocks) {
        if (y < 0 || y >= CHUNK_HEIGHT) return false;
        const key = blockKey(x, y, z);
        if (editedBlocks && key in editedBlocks) return editedBlocks[key];
        return y <= getHeight(x, z);
    }
    // Helper: returns true only if block (x,y,z) is natural and unmodified by the player
    function isNaturalSolidBlock(x, y, z, editedBlocks) {
        if (y < 0 || y >= CHUNK_HEIGHT) return false;
        const key = blockKey(x, y, z);
        if (editedBlocks && key in editedBlocks) return false; // user modified – not natural
        return y <= getHeight(x, z);
    }
    function createFace(x, y, z, normal) {
        return {pos: [x, y, z], normal: normal};
    }

    onmessage = function(e) {
        const { chunkX, chunkZ, chunkSize, chunkHeight, editedBlocks } = e.data;
        CHUNK_SIZE = chunkSize;
        CHUNK_HEIGHT = chunkHeight;

        const faces = [];
        const startX = chunkX * CHUNK_SIZE;
        const startZ = chunkZ * CHUNK_SIZE;

        for (let x = 0; x < CHUNK_SIZE; x++) {
            for (let z = 0; z < CHUNK_SIZE; z++) {
                const worldX = startX + x;
                const worldZ = startZ + z;
                let height = CHUNK_HEIGHT - 1;
                while (height >= 0 && !hasBlock(worldX, height, worldZ, editedBlocks)) height--;
                for (let y = 0; y <= height; y++) {
                    if (!hasBlock(worldX, y, worldZ, editedBlocks)) continue;
                    const faceDirs = [
                        [ 0,  1,  0], [ 0, -1,  0],
                        [ 1,  0,  0], [-1,  0,  0],
                        [ 0,  0,  1], [ 0,  0, -1]
                    ];
                    const currIsNatural = isNaturalSolidBlock(worldX, y, worldZ, editedBlocks);
                    for (const normal of faceDirs) {
                        const nx = worldX + normal[0],
                            ny = y      + normal[1],
                            nz = worldZ + normal[2];
                        const neighborExists = hasBlock(nx, ny, nz, editedBlocks);
                        const neighborIsNatural = isNaturalSolidBlock(nx, ny, nz, editedBlocks);
                        // Draw face if neighbor is air OR either block was modified by player
                        if (!neighborExists || !(currIsNatural && neighborIsNatural)) {
                            faces.push(createFace(worldX, y, worldZ, normal));
                        }
                    }
                }
            }
        }
        postMessage({faces});
    };
    `
		],
		{ type: "application/javascript" }
	);
}
if (!chunkWorkerBlobURL) {
	chunkWorkerBlobURL = URL.createObjectURL(getChunkWorkerBlob());
}

// Create pool of workers
for (let i = 0; i < WORKER_POOL_SIZE; ++i) {
	const worker = new Worker(chunkWorkerBlobURL);
	workerPool.push(worker);
	chunkPool.push({ worker, busy: false });
}
function getAvailableWorker() {
	if (outstandingJobs < maxConcurrentWorkers)
		return chunkPool.find((p) => !p.busy);
	return null;
}

function tryDispatchQueuedJobs() {
	while (chunkJobQueue.length && outstandingJobs < maxConcurrentWorkers) {
		const available = getAvailableWorker();
		if (!available) break;
		const {
			chunkX,
			chunkZ,
			chunkKey,
			resolve,
			reject,
			editObj
		} = chunkJobQueue.shift();
		performChunkJob(
			chunkX,
			chunkZ,
			chunkKey,
			resolve,
			reject,
			editObj,
			available
		);
	}
}
function performChunkJob(
	chunkX,
	chunkZ,
	chunkKey,
	resolve,
	reject,
	editObj,
	poolWorker
) {
	outstandingJobs++;
	const t0 = performance.now();
	poolWorker.busy = true;
	poolWorker.worker.onmessage = function (event) {
		outstandingJobs--;
		poolWorker.busy = false;
		updateJobTimings(performance.now() - t0);
		resolve(event.data.faces);
		setTimeout(tryDispatchQueuedJobs, 0);
	};
	poolWorker.worker.onerror = function (err) {
		outstandingJobs--;
		poolWorker.busy = false;
		reject(err);
		setTimeout(tryDispatchQueuedJobs, 0);
	};
	poolWorker.worker.postMessage({
		chunkX,
		chunkZ,
		chunkSize: CHUNK_SIZE,
		chunkHeight: CHUNK_HEIGHT,
		editedBlocks: editObj
	});
}

// --- Main: Overwrite requestChunkGeometry with queue-aware system
function requestChunkGeometry(chunkX, chunkZ, chunkKey) {
	return new Promise((resolve, reject) => {
		// Collect edits from this chunk and neighbors — so culling at boundaries can see edits
		const editObj = {};
		for (const [dx, dz] of [
			[0, 0],
			[1, 0],
			[-1, 0],
			[0, 1],
			[0, -1]
		]) {
			const key = `${chunkX + dx},${chunkZ + dz}`;
			const edits = chunkEdits.get(key);
			if (edits) Object.assign(editObj, edits);
		}
		chunkJobQueue.push({ chunkX, chunkZ, chunkKey, resolve, reject, editObj });
		tryDispatchQueuedJobs();
	});
}

// Store user-placed/removed blocks "edits" for each chunk, keys: blockKey(x,y,z) -> Boolean
const chunkEdits = new Map();
function getChunkEditObj(chunkX, chunkZ) {
	const key = `${chunkX},${chunkZ}`;
	return chunkEdits.get(key) || {};
}
function recordChunkEdit(x, y, z, present) {
	const chunkX = Math.floor(x / CHUNK_SIZE);
	const chunkZ = Math.floor(z / CHUNK_SIZE);
	const editKey = `${chunkX},${chunkZ}`;
	if (!chunkEdits.has(editKey)) chunkEdits.set(editKey, {});
	chunkEdits.get(editKey)[blockKey(x, y, z)] = present;
}

// --- Geometry Conversion (Worker output to BufferGeometry on main thread) --
function facesToBufferGeometry(faces) {
	const faceGeos = [];
	for (let f of faces) {
		const {
			pos: [x, y, z],
			normal: [nx, ny, nz]
		} = f;
		const plane = new THREE.PlaneGeometry(1, 1);
		const quat = new THREE.Quaternion();
		quat.setFromUnitVectors(
			new THREE.Vector3(0, 0, 1),
			new THREE.Vector3(nx, ny, nz)
		);
		plane.applyQuaternion(quat);
		plane.translate(x + 0.5 + nx * 0.5, y + 0.5 + ny * 0.5, z + 0.5 + nz * 0.5);
		faceGeos.push(plane);
	}
	if (!faceGeos.length) return null;
	// Merge individual face planes and weld vertices to reduce redundant geometry
	let geometry = mergeBufferGeometries(faceGeos, false);
	geometry = mergeVertices(geometry);
	geometry.computeBoundingSphere();
	return geometry;
}

// --- Mobile Controls ---
const isMobile = "ontouchstart" in window || navigator.maxTouchPoints > 0;
let mobileDirection = { x: 0, y: 0 };
let mobileJump = false;
if (isMobile && window.nipplejs) {
	const joystick = nipplejs.create({
		zone: document.getElementById("joystickZone"),
		mode: "static",
		position: { left: "60px", bottom: "60px" },
		color: "white"
	});
	joystick.on("move", (_evt, data) => {
		mobileDirection.x = data.vector.x;
		mobileDirection.y = data.vector.y;
	});
	joystick.on("end", () => {
		mobileDirection.x = 0;
		mobileDirection.y = 0;
	});
	document.getElementById("jumpBtn").addEventListener("touchstart", (e) => {
		e.preventDefault();
		mobileJump = true;
	});
	document.getElementById("removeBtn").addEventListener("touchstart", (e) => {
		e.preventDefault();
		if (!isPaused && highlightBlock) {
			setBlock(...highlightBlock, false);
			highlightBlock = null;
		}
	});
	document.getElementById("placeBtn").addEventListener("touchstart", (e) => {
		e.preventDefault();
		if (!isPaused) {
			const targeted = getTargetedBlock();
			if (targeted && targeted.placePos) {
				const [px, py, pz] = targeted.placePos;
				setBlock(px, py, pz, true);
				highlightBlock = null;
			}
		}
	});
	// Touch-to-look on right half
	let lookTouchId = null,
		lastX = 0,
		lastY = 0;
	document.addEventListener("touchstart", (e) => {
		for (let t of e.changedTouches) {
			if (
				t.clientX > window.innerWidth / 2 &&
				!t.target.closest("#mobileControls")
			) {
				lookTouchId = t.identifier;
				lastX = t.clientX;
				lastY = t.clientY;
			}
		}
	});
	document.addEventListener("touchmove", (e) => {
		for (let t of e.changedTouches) {
			if (t.identifier === lookTouchId) {
				const dx = t.clientX - lastX,
					dy = t.clientY - lastY;
				lastX = t.clientX;
				lastY = t.clientY;
				camera.rotation.y -= dx * 0.002;
				camera.rotation.x -= dy * 0.002;
				camera.rotation.x = Math.max(
					-Math.PI / 2,
					Math.min(Math.PI / 2, camera.rotation.x)
				);
			}
		}
	});
	document.addEventListener("touchend", (e) => {
		for (let t of e.changedTouches) {
			if (t.identifier === lookTouchId) {
				lookTouchId = null;
			}
		}
	});
}

// --- Animation and Main Loop ---
let lastFrameTime = 0;
function animate(now) {
	requestAnimationFrame(animate);

	// -- Cap to display refresh rate: --
	// "now" is a high-resolution timestamp provided by requestAnimationFrame
	// Don't render more than 1 frame per ~16.7ms (for 60Hz) or ~8.33ms (for 120Hz);
	// But requestAnimationFrame already only triggers once per display refresh!
	// So to "cap" to refresh, just call requestAnimationFrame *once* per render.
	// We also ensure no internal setTimeout/setInterval are used for rendering.

	if (!isPaused) {
		// movement vectors
		camera.getWorldDirection(forwardVec);
		forwardVec.y = 0;
		forwardVec.normalize();
		rightVec.crossVectors(forwardVec, camera.up).normalize();

		moveDirVec.set(0, 0, 0);
		if (keys.forward) moveDirVec.add(forwardVec);
		if (keys.backward) moveDirVec.sub(forwardVec);
		if (keys.right) moveDirVec.add(rightVec);
		if (keys.left) moveDirVec.sub(rightVec);
		// mobile joystick adds
		if (isMobile) {
			if (mobileDirection.y)
				moveDirVec.add(forwardVec.clone().multiplyScalar(mobileDirection.y));
			if (mobileDirection.x)
				moveDirVec.add(rightVec.clone().multiplyScalar(mobileDirection.x));
		}

		if (moveDirVec.lengthSq() > 0) {
			moveDirVec.normalize().multiplyScalar(moveSpeed);
			// X axis
			tempPosVec.copy(camera.position).addScaledVector(moveDirVec, 1);
			tempPosVec.z = camera.position.z;
			if (!checkCollision(tempPosVec)) camera.position.x = tempPosVec.x;
			// Z axis
			tempPosVec.copy(camera.position).addScaledVector(moveDirVec, 1);
			tempPosVec.x = camera.position.x;
			if (!checkCollision(tempPosVec)) camera.position.z = tempPosVec.z;
		}

		// Jump: keyboard or mobile
		if ((keys.jump || (isMobile && mobileJump)) && onGround) {
			velocity.y = jumpStrength;
			onGround = false;
			mobileJump = false;
		}

		velocity.y -= gravity;
		// Y axis
		const newY = camera.position.y + velocity.y;
		tempPosVec.copy(camera.position);
		tempPosVec.y = newY;
		if (!checkCollision(tempPosVec)) {
			camera.position.y = newY;
			onGround = false;
		} else {
			if (velocity.y > 0) {
				velocity.y = 0;
			} else {
				const footY = newY - playerHeight;
				const blockY = Math.floor(footY + 1e-4);
				camera.position.y = blockY + 1 + playerHeight;
				velocity.y = 0;
				onGround = true;
			}
		}

		// NEW: Get target block with adjacent air for placement
		const targeted = getTargetedBlock();
		if (targeted) {
			highlightBlock = targeted.pos;
			highlightNormal = targeted.normal;
			highlightMesh.visible = true;
			highlightMesh.position.set(...highlightBlock).addScalar(0.5);
		} else {
			highlightMesh.visible = false;
			highlightBlock = null;
			highlightNormal = null;
		}

		// Only rebuild chunk set when player crosses chunk boundary
		const camChunkX = Math.floor(camera.position.x / CHUNK_SIZE);
		const camChunkZ = Math.floor(camera.position.z / CHUNK_SIZE);
		if (camChunkX !== lastCamChunkX || camChunkZ !== lastCamChunkZ) {
			lastCamChunkX = camChunkX;
			lastCamChunkZ = camChunkZ;
			updateChunks();
		}
	} else {
		// When paused, hide highlight/crosshair
		highlightMesh.visible = false;
	}

	// Frustum culling: hide chunks not in camera view
	camera.updateMatrixWorld();
	camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
	projScreenMatrix.multiplyMatrices(
		camera.projectionMatrix,
		camera.matrixWorldInverse
	);
	frustum.setFromProjectionMatrix(projScreenMatrix);
	for (const mesh of chunks.values()) {
		const bs = mesh.geometry?.boundingSphere;
		mesh.visible = bs ? frustum.intersectsSphere(bs) : true;
	}

	// Occlusion culling
	if (ext) {
		gl.colorMask(false, false, false, false);
		gl.depthMask(false);
		for (const mesh of chunks.values()) {
			if (!mesh.userData.query || !mesh.visible) continue;
			gl.beginQuery(ext.ANY_SAMPLES_PASSED_CONSERVATIVE, mesh.userData.query);
			renderer.renderBufferDirect(
				camera,
				scene.fog,
				mesh.userData.boundingBoxMesh.geometry,
				boundingBoxMaterial,
				mesh.userData.boundingBoxMesh,
				null
			);
			gl.endQuery(ext.ANY_SAMPLES_PASSED_CONSERVATIVE);
		}
		gl.colorMask(true, true, true, true);
		gl.depthMask(true);
		for (const mesh of chunks.values()) {
			if (!mesh.userData.query) continue;
			if (gl.getQueryParameter(mesh.userData.query, gl.QUERY_RESULT_AVAILABLE)) {
				mesh.visible = Boolean(
					gl.getQueryParameter(mesh.userData.query, gl.QUERY_RESULT)
				);
			}
		}
	}

	// Step sound effects
	if (moveDirVec.lengthSq() > 0 && onGround) {
		stepTimer += (now - lastFrameTime) / 1000;
		if (stepTimer > 0.4) {
			playSound("step");
			stepTimer = 0;
		}
	}

	// Update skybox position to follow camera
	skybox.position.copy(camera.position);

	// Update floating items
	updateItems((now - lastFrameTime) / 1000);

	renderer.render(scene, camera);
	lastFrameTime = now;
}

// --- Setup ---
const startX = 0,
	startZ = 30;
const groundAtStart = getGroundHeight(startX, startZ);
function findSafeSpawnY(x, z, height) {
	let y = Math.max(getGroundHeight(x, z), height + 1);
	const maxY = CHUNK_HEIGHT + 8;
	outer: for (; y < maxY; y++) {
		for (let dy = 0; dy < height; dy++) {
			if (hasBlock(Math.floor(x), Math.floor(y - dy - 1e-4), Math.floor(z))) {
				continue outer;
			}
		}
		return y;
	}
	return maxY;
}
const safeY = findSafeSpawnY(startX, startZ, playerHeight);
camera.position.set(startX, safeY, startZ);

// --- Enhanced Terrain & Biomes ---
const BIOME_TYPES = {
	FOREST: { color: 0x228b22, name: "Forest" },
	DESERT: { color: 0xffd700, name: "Desert" },
	SNOW: { color: 0xffffff, name: "Snow" },
	STONE: { color: 0x696969, name: "Stone" },
	WATER: { color: 0x4169e1, name: "Water" }
};

function getBiome(x, z) {
	const temp = simplex.noise2D(x * 0.003, z * 0.003);
	const humidity = simplex.noise2D(x * 0.004 + 1000, z * 0.004 + 1000);

	if (temp < -0.3) return BIOME_TYPES.SNOW;
	if (humidity < -0.2) return BIOME_TYPES.DESERT;
	if (temp > 0.4 && humidity < 0.1) return BIOME_TYPES.STONE;
	return BIOME_TYPES.FOREST;
}

function getBlockMaterial(x, y, z) {
	const biome = getBiome(x, z);
	const height = getHeight(x, z);

	if (y <= height - 5) return blockMaterials.stone;
	if (y === height && biome === BIOME_TYPES.DESERT) return blockMaterials.sand;
	if (y === height && biome === BIOME_TYPES.SNOW) return blockMaterials.snow;
	if (y < height) return blockMaterials.stone;
	return blockMaterials.grass;
}

// --- Skybox Setup ---
function createSkybox() {
	const skyboxGeometry = new THREE.BoxGeometry(800, 800, 800);
	const skyboxMaterials = [
		new THREE.MeshBasicMaterial({ color: 0x87ceeb }), // right
		new THREE.MeshBasicMaterial({ color: 0x87ceeb }), // left
		new THREE.MeshBasicMaterial({ color: 0x87ceeb }), // top
		new THREE.MeshBasicMaterial({ color: 0x4682b4 }), // bottom
		new THREE.MeshBasicMaterial({ color: 0x87ceeb }), // front
		new THREE.MeshBasicMaterial({ color: 0x87ceeb }) // back
	];
	skyboxMaterials.forEach((material) => (material.side = THREE.BackSide));
	const skybox = new THREE.Mesh(skyboxGeometry, skyboxMaterials);
	scene.add(skybox);
	return skybox;
}

const skybox = createSkybox();

// --- Audio System ---
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const sounds = {};

function createAudioBuffer(frequency, duration, type = "sine") {
	const sampleRate = audioContext.sampleRate;
	const frameCount = sampleRate * duration;
	const buffer = audioContext.createBuffer(1, frameCount, sampleRate);
	const channelData = buffer.getChannelData(0);

	for (let i = 0; i < frameCount; i++) {
		const t = i / sampleRate;
		const envelope = Math.exp(-t * 3);

		let sample = 0;
		if (type === "sine") {
			sample = Math.sin(2 * Math.PI * frequency * t) * envelope;
		} else if (type === "noise") {
			sample = (Math.random() * 2 - 1) * envelope * 0.3;
		}
		channelData[i] = sample * 0.3;
	}
	return buffer;
}

// Initialize sound effects
sounds.place = createAudioBuffer(440, 0.2);
sounds.break = createAudioBuffer(220, 0.3, "noise");
sounds.pickup = createAudioBuffer(660, 0.4);
sounds.step = createAudioBuffer(150, 0.1, "noise");

function playSound(soundName) {
	if (!sounds[soundName]) return;
	const source = audioContext.createBufferSource();
	const gainNode = audioContext.createGain();
	source.buffer = sounds[soundName];
	source.connect(gainNode);
	gainNode.connect(audioContext.destination);
	gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
	source.start();
}

// --- Item System ---
const ITEM_TYPES = {
	WOOD: { name: "Wood", color: 0x8b4513, symbol: "🪵" },
	STONE: { name: "Stone", color: 0x696969, symbol: "🪨" },
	CRYSTAL: { name: "Crystal", color: 0xff69b4, symbol: "💎" },
	BERRY: { name: "Berry", color: 0xff0000, symbol: "🍓" }
};

const inventory = {
	slots: new Array(8).fill(null).map(() => ({ type: null, count: 0 })),
	selectedSlot: 0
};

const worldItems = new Map(); // worldX,worldY,worldZ -> {type, mesh}

function spawnItem(x, y, z, itemType) {
	const key = `${x},${y},${z}`;
	if (worldItems.has(key)) return;

	const geometry = new THREE.SphereGeometry(0.2, 8, 6);
	const material = new THREE.MeshStandardMaterial({
		color: itemType.color,
		emissive: itemType.color,
		emissiveIntensity: 0.2
	});
	const mesh = new THREE.Mesh(geometry, material);
	mesh.position.set(x + 0.5, y + 0.5, z + 0.5);

	// Floating animation
	mesh.userData.startY = y + 0.5;
	mesh.userData.time = Math.random() * Math.PI * 2;

	scene.add(mesh);
	worldItems.set(key, { type: itemType, mesh });
}

function updateItems(deltaTime) {
	for (const [key, item] of worldItems.entries()) {
		item.mesh.userData.time += deltaTime * 2;
		item.mesh.position.y =
			item.mesh.userData.startY + Math.sin(item.mesh.userData.time) * 0.1;
		item.mesh.rotation.y += deltaTime;

		// Check for pickup
		const dist = camera.position.distanceTo(item.mesh.position);
		if (dist < 2) {
			collectItem(key, item.type);
		}
	}
}

function collectItem(itemKey, itemType) {
	// Find empty slot or stack
	let targetSlot = inventory.slots.find(
		(slot) => slot.type === itemType.name && slot.count < 64
	);

	if (!targetSlot) {
		targetSlot = inventory.slots.find((slot) => slot.type === null);
	}

	if (targetSlot) {
		if (targetSlot.type === null) {
			targetSlot.type = itemType.name;
			targetSlot.count = 1;
		} else {
			targetSlot.count++;
		}

		// Remove from world
		const item = worldItems.get(itemKey);
		scene.remove(item.mesh);
		worldItems.delete(itemKey);

		updateInventoryUI();
		showItemPickup(itemType.name);
		playSound("pickup");
	}
}

function updateInventoryUI() {
	const inventoryEl = document.getElementById("inventory");
	inventoryEl.innerHTML = "";

	inventory.slots.forEach((slot, index) => {
		const slotEl = document.createElement("div");
		slotEl.className = `inventory-slot ${
			index === inventory.selectedSlot ? "selected" : ""
		}`;

		if (slot.type) {
			const itemType = Object.values(ITEM_TYPES).find((t) => t.name === slot.type);
			slotEl.innerHTML = `${itemType?.symbol || "?"}<span class="count">${
				slot.count
			}</span>`;
		}

		inventoryEl.appendChild(slotEl);
	});
}

function showItemPickup(itemName) {
	const el = document.getElementById("itemPickupText");
	el.textContent = `+1 ${itemName}`;
	el.style.opacity = "1";
	setTimeout(() => {
		el.style.opacity = "0";
	}, 1000);
}

// Randomly spawn items in chunks
function spawnItemsInChunk(chunkX, chunkZ) {
	const startX = chunkX * CHUNK_SIZE;
	const startZ = chunkZ * CHUNK_SIZE;

	for (let i = 0; i < 3; i++) {
		const x = startX + Math.floor(Math.random() * CHUNK_SIZE);
		const z = startZ + Math.floor(Math.random() * CHUNK_SIZE);
		const y = getHeight(x, z) + 1;

		const biome = getBiome(x, z);
		let itemType;

		if (biome === BIOME_TYPES.FOREST) itemType = ITEM_TYPES.WOOD;
		else if (biome === BIOME_TYPES.DESERT) itemType = ITEM_TYPES.CRYSTAL;
		else if (biome === BIOME_TYPES.SNOW) itemType = ITEM_TYPES.BERRY;
		else itemType = ITEM_TYPES.STONE;

		if (Math.random() < 0.3) {
			spawnItem(x, y, z, itemType);
		}
	}
}

// Enhanced controls
function dropItem() {
	const slot = inventory.slots[inventory.selectedSlot];
	if (slot.count > 0) {
		const itemType = Object.values(ITEM_TYPES).find((t) => t.name === slot.type);
		const dropPos = camera.position
			.clone()
			.add(camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(2));

		spawnItem(
			Math.floor(dropPos.x),
			Math.floor(dropPos.y),
			Math.floor(dropPos.z),
			itemType
		);

		slot.count--;
		if (slot.count === 0) {
			slot.type = null;
		}
		updateInventoryUI();
	}
}

function collectNearbyItems() {
	for (const [key, item] of worldItems.entries()) {
		const dist = camera.position.distanceTo(item.mesh.position);
		if (dist < 3) {
			collectItem(key, item.type);
			break;
		}
	}
}

// --- Audio system for enhanced movement ---
let stepTimer = 0;
let lastGroundType = null;

// --- Resize Handler ---
window.addEventListener("resize", () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

// Reusable vectors to reduce per-frame allocations
const forwardVec = new THREE.Vector3();
const rightVec = new THREE.Vector3();
const moveDirVec = new THREE.Vector3();
const tempPosVec = new THREE.Vector3();

// Track player's current chunk to avoid rebuilding every frame
let lastCamChunkX = null,
	lastCamChunkZ = null;

// --- Mouse Controls for block manip
renderer.domElement.addEventListener("mousedown", (e) => {
	if (!controls.isLocked || isPaused) return;
	if (!highlightBlock || !highlightNormal) return;

	// Remove block (LEFT CLICK, button 0)
	if (e.button === 0) {
		const [x, y, z] = highlightBlock;
		if (
			!(
				Math.floor(camera.position.x) === x &&
				Math.floor(camera.position.y - playerHeight / 2) === y &&
				Math.floor(camera.position.z) === z
			)
		) {
			setBlock(x, y, z, false);
			highlightBlock = null;
			playSound("break");

			// Chance to drop items when breaking blocks
			if (Math.random() < 0.2) {
				const biome = getBiome(x, z);
				let itemType = ITEM_TYPES.STONE;
				if (biome === BIOME_TYPES.FOREST) itemType = ITEM_TYPES.WOOD;
				spawnItem(x, y + 1, z, itemType);
			}
		}
	}
	// Place block (RIGHT CLICK, button 2)
	if (e.button === 2) {
		const targeted = getTargetedBlock();
		if (targeted && targeted.placePos) {
			const [px, py, pz] = targeted.placePos;
			if (py >= 0 && py < CHUNK_HEIGHT) {
				function wouldPlayerBeObstructedByBlockAt(x, y, z) {
					// ... existing code ...
				}
				if (!wouldPlayerBeObstructedByBlockAt(px, py, pz)) {
					setBlock(px, py, pz, true);
					highlightBlock = null;
					playSound("place");
				}
			}
		}
	}
});
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

// Initialize inventory UI on load
const inventoryEl = document.createElement("div");
inventoryEl.id = "inventory";
document.body.appendChild(inventoryEl);
updateInventoryUI();

requestAnimationFrame(animate);
