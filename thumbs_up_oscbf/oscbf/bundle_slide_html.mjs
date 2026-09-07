import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";


const root = path.dirname(fileURLToPath(import.meta.url));
const figureDir = path.join(
  root, "outputs", "a4_joint_limit_comparison", "result_figures"
);
const source = path.join(figureDir, "OSCBF_A4限位与避障_单页汇报.html");
const output = path.join(figureDir, "OSCBF_A4限位与避障_单页汇报_自包含.html");

let html = fs.readFileSync(source, "utf8");
for (const filename of [
  "04_末端轨迹与结果汇总.png",
  "05_带障碍物场景_A4关节轨迹.png",
]) {
  const bytes = fs.readFileSync(path.join(figureDir, filename));
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  html = html.replace(`src="${filename}"`, `src="${dataUrl}"`);
}

fs.writeFileSync(output, html, "utf8");
const required = [
  "−0.500 → ≈0 rad",
  "1537 → 0",
  "min h = 0.000121 m",
  "1.56 cm",
  "data:image/png;base64,",
];
for (const token of required) {
  if (!html.includes(token)) throw new Error(`Missing required content: ${token}`);
}
if ((html.match(/data:image\/png;base64,/g) || []).length !== 2) {
  throw new Error("Expected exactly two embedded PNG images");
}

console.log(JSON.stringify({ output, bytes: fs.statSync(output).size, embeddedImages: 2 }));
