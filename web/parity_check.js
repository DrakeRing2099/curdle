// Run the JavaScript engine for N simulated years and print KPI distributions,
// to compare with the Python engine (web/parity_python.py prints the same KPIs).
//   node web/parity_check.js 30 Habit
const fs = require("fs");
if (fs.existsSync(__dirname + "/policy.json")) globalThis.DAIRY_POLICY = JSON.parse(fs.readFileSync(__dirname + "/policy.json"));
const { Twin, P } = require("./engine.js");
const cfg = require("./config.json");
const N = +(process.argv[2] || 20), policy = process.argv[3] || "Habit";
const severe = process.argv[4] === "severe";

function kpis(seed) {
  const scen = {};
  if (severe) { scen.forced_weather = {}; for (let w = 16; w < 22; w++) scen.forced_weather[w] = 1; }
  const tw = new Twin(cfg, scen, policy, seed);
  const start = tw.unitsInSystem();
  for (let d = 0; d < cfg.days; d++) tw.step();
  const L = tw.log;
  let D = 0, S = 0, Dh = 0, Sh = 0, waste = 0, made = 0;
  const madeP = [0, 0, 0, 0], salesP = [0, 0, 0, 0], wasteP = [0, 0, 0, 0];
  const wo = [], wd = [];
  for (const r of L) {
    for (let c = 0; c < 3; c++) for (let p = 0; p < P; p++) {
      D += r.demand[c][p]; S += r.sales[c][p]; salesP[p] += r.sales[c][p];
      if (r.weather) { Dh += r.demand[c][p]; Sh += r.sales[c][p]; }
    }
    for (let p = 0; p < P; p++) {
      const w = r.wasteK[p] + r.wasteP[p] + r.wasteM[p] + r.wasteQ[p];
      waste += w; wasteP[p] += w; made += r.made[p]; madeP[p] += r.made[p];
    }
    const wk = Math.floor(r.day / 7);
    wo[wk] = (wo[wk] || 0) + r.kOrders[0]; wd[wk] = (wd[wk] || 0) + r.demand[0][0];
  }
  const end = tw.unitsInSystem();
  for (let p = 0; p < P; p++) {
    const gap = start[p] + madeP[p] - salesP[p] - wasteP[p] - end[p];
    if (gap !== 0) throw new Error(`conservation broken: product ${p} gap ${gap}`);
  }
  const v = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length; };
  return { fill: S / D, fill_heat: Dh ? Sh / Dh : NaN, waste_share: waste / made, lost: D - S,
           bullwhip: v(wo) / v(wd), migrated: cfg.city.households[0] - L[L.length - 1].customers[0],
           profit: L[L.length - 1].cash - cfg.money.opening_cash };
}

const t0 = Date.now(), rows = [];
for (let s = 0; s < N; s++) rows.push(kpis(s));
const keys = Object.keys(rows[0]), out = {};
for (const k of keys) {
  const a = rows.map(r => r[k]).filter(x => !Number.isNaN(x)).sort((x, y) => x - y);
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1));
  out[k] = { mean: m, sd, p5: a[Math.floor(0.05 * (a.length - 1))], p95: a[Math.ceil(0.95 * (a.length - 1))] };
}
console.log(JSON.stringify({ engine: "js", policy, severe, N, sec_per_year: (Date.now() - t0) / 1000 / N, kpis: out }, null, 1));
