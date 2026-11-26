import fs from "node:fs";
import path from "node:path";

const version = process.argv[2];
if (!version || !/^v\d+$/.test(version)) {
  console.error("Usage: npm run snapshot -- v<number> (e.g. v1, v2)");
  process.exit(1);
}

const rootDir = process.cwd();
const distDir = path.join(rootDir, "dist");
const targetDir = path.join(rootDir, "public", version);

const distIndex = path.join(distDir, "index.html");
const distAssets = path.join(distDir, "assets");

if (!fs.existsSync(distIndex) || !fs.existsSync(distAssets)) {
  console.error("Missing dist/index.html or dist/assets/. Run `npm run build` first.");
  process.exit(1);
}

fs.rmSync(targetDir, { recursive: true, force: true });
fs.mkdirSync(targetDir, { recursive: true });
fs.copyFileSync(distIndex, path.join(targetDir, "index.html"));
fs.cpSync(distAssets, path.join(targetDir, "assets"), { recursive: true });

console.log(`Archived build snapshot to public/${version}/ (index.html + assets/)`);
