/* Dairy twin engine, JavaScript port of the Python package (dairytwin/).
 * Same rules, same parameters (config.json exported from Python), same daily order of
 * events. Random number generators differ, so the two engines agree in distribution,
 * not run for run (checked by web/parity_check.js).
 */
(function (root) {
  "use strict";
  const P = 4, MILK = 0, CURD = 1, PANEER = 2, BUTTER = 3;
  const KIR = 0, MT = 1, QC = 2;
  const P_SPILL = 0.30;

  // ------------------------------------------------------------------ random numbers
  class RNG {
    constructor(seed) {
      let s = (seed >>> 0) ^ 0x9e3779b9;
      this.a = s; this.b = s ^ 0x243f6a88; this.c = s ^ 0xb7e15162; this.d = s ^ 0x7f4a7c15;
      for (let i = 0; i < 15; i++) this.random();
      this.spare = null;
    }
    random() { // sfc32
      let a = this.a >>> 0, b = this.b >>> 0, c = this.c >>> 0, d = this.d >>> 0;
      const t = (a + b + d) >>> 0;
      d = (d + 1) >>> 0; a = b ^ (b >>> 9); b = (c + (c << 3)) >>> 0;
      c = (c << 21) | (c >>> 11); c = (c + t) >>> 0;
      this.a = a; this.b = b; this.c = c; this.d = d;
      return t / 4294967296;
    }
    normal() {
      if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
      let u, v, s;
      do { u = 2 * this.random() - 1; v = 2 * this.random() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const m = Math.sqrt(-2 * Math.log(s) / s);
      this.spare = v * m; return u * m;
    }
    uniform(lo, hi) { return lo + (hi - lo) * this.random(); }
    lognormal(mu, sigma) { return Math.exp(mu + sigma * this.normal()); }
    poisson(lam) {
      if (lam <= 0) return 0;
      if (lam < 30) {
        const L = Math.exp(-lam); let k = 0, p = 1;
        do { k++; p *= this.random(); } while (p > L);
        return k - 1;
      }
      // PTRS (Hormann 1993) transformed rejection
      const slam = Math.sqrt(lam), loglam = Math.log(lam);
      const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b, invalpha = 1.1239 + 1.1328 / (b - 3.4);
      const vr = 0.9277 - 3.6224 / (b - 2);
      for (;;) {
        const U = this.random() - 0.5, V = this.random(), us = 0.5 - Math.abs(U);
        const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
        if (us >= 0.07 && V <= vr) return k;
        if (k < 0 || (us < 0.013 && V > us)) continue;
        if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - lgamma(k + 1)) return k;
      }
    }
    binomial(n, p) {
      if (n <= 0 || p <= 0) return 0;
      if (p >= 1) return n;
      if (n < 40) { let k = 0; for (let i = 0; i < n; i++) if (this.random() < p) k++; return k; }
      const mean = n * p, sd = Math.sqrt(n * p * (1 - p));
      if (sd < 3) { // small variance: Poisson-like via inversion on Bernoulli sums is slow; use Poisson approx
        const k = p < 0.5 ? this.poisson(mean) : n - this.poisson(n * (1 - p));
        return Math.max(0, Math.min(n, k));
      }
      return Math.max(0, Math.min(n, Math.round(mean + sd * this.normal())));
    }
    permutation(n) {
      const a = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(this.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
      return a;
    }
    dirichlet(alpha, n) { // symmetric, via gamma(alpha) with alpha >= 1 (Marsaglia-Tsang)
      const g = [];
      for (let i = 0; i < n; i++) g.push(this.gamma(alpha));
      const s = g.reduce((x, y) => x + y, 0); return g.map(x => x / s);
    }
    gamma(k) {
      const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = this.normal(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v; const u = this.random();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    }
  }
  function lgamma(x) { // Stirling series, x >= 1
    if (x < 7) { let r = 0; while (x < 7) { r -= Math.log(x); x += 1; } return r + lgamma(x); }
    const x2 = 1 / (x * x);
    return (x - 0.5) * Math.log(x) - x + 0.918938533204673 + (1 / 12 - x2 * (1 / 360 - x2 / 1260)) / x;
  }

  // ------------------------------------------------------------------ age-structured stock
  // a "slot" is one product's age profile: stock[off .. off + A)
  function slotTotal(s, off, A) { let t = 0; for (let a = 0; a < A; a++) t += s[off + a]; return t; }
  function slotTake(s, off, A, qty, into, intoOff) { // oldest first; returns units taken
    let left = qty;
    for (let a = A - 1; a >= 0 && left > 0; a--) {
      const g = Math.min(s[off + a], left);
      if (g > 0) { s[off + a] -= g; left -= g; if (into) into[intoOff + a] += g; }
    }
    return qty - left;
  }
  function slotAge(s, off, A, shelf, shift) { // age by shift days; return expired units
    let expired = 0;
    if (shift > 0) {
      for (let a = A - 1; a >= 0; a--) {
        const v = s[off + a]; if (!v) continue;
        s[off + a] = 0;
        const na = a + shift;
        if (na >= shelf || na >= A) expired += v; else s[off + na] += v;
      }
    } else {
      for (let a = shelf; a < A; a++) { expired += s[off + a]; s[off + a] = 0; }
    }
    return expired;
  }

  // ------------------------------------------------------------------ the twin
  class Twin {
    constructor(cfg, scen = {}, policy = "Habit", seed = 0, citySeed = 0) {
      this.cfg = cfg; this.policyName = policy; this.seed = seed;
      this.scen = Object.assign({ battery_fridge_share: 0, battery_targeting: "risk", kirana_returns: false,
        outages_enabled: true, forced_weather: {}, forced_outages: {}, force_twice_daily: null }, scen);
      const names = ["regimes", "supply", "kirana", "distributor", "mt", "qc", "power", "consumers"];
      this.rng = {}; names.forEach((n, i) => { this.rng[n] = new RNG(seed * 7919 + i * 104729 + 17); });
      this.rng.city = new RNG(citySeed * 31337 + 2026); // the fixed city: same in every year
      const pr = cfg.products;
      this.A = cfg.max_age;
      this.shelf = pr.shelf_life; this.vol = pr.fridge_volume; this.pc = pr.per_capita;
      this.C = cfg.city.households.slice();
      this.migrated = 0;
      this.base = [0, 1, 2].map(c => this.pc.map(x => this.C[c] * x));
      this.buildCity();
      this.buildKirana();
      this.buildChannels();
      this.regime = { weather: 0, supply: 0, festival: 0 };
      this.cash = cfg.money.opening_cash;
      this.day = 0; this.lastMult = [1, 1, 1, 1];
      this.orderHist = [];
      const tot = [0, 0, 0, 0];
      for (let p = 0; p < P; p++) tot[p] = this.base[0][p] + this.base[1][p] + this.base[2][p];
      for (let i = 0; i < 7; i++) this.orderHist.push(tot.slice());
      this.log = [];
      this.watch = -1; this.watchLog = [];
    }

    // ---- city
    buildCity() {
      const r = this.rng.city, cfg = this.cfg, [gx, gy] = cfg.city.grid, nLoc = gx * gy;
      this.nLoc = nLoc; this.gx = gx; this.gy = gy;
      const w = r.dirichlet(6, nLoc), n = cfg.city.n_kirana;
      const cum = []; w.reduce((s, x, i) => (cum[i] = s + x), 0);
      const loc = [];
      for (let i = 0; i < n; i++) { const u = r.random(); let l = cum.findIndex(c => u <= c); if (l < 0) l = nLoc - 1; loc.push(l); }
      loc.sort((a, b) => a - b);
      this.loc = loc;
      this.pos = loc.map(l => [l % gx + r.uniform(0.08, 0.92), Math.floor(l / gx) + r.uniform(0.30, 0.94)]);
      this.route = loc.map(l => Math.floor(l * cfg.city.n_routes / nLoc));
      let fr = []; for (let i = 0; i < nLoc; i++) fr.push(r.lognormal(0, 0.35));
      const m = fr.reduce((a, b) => a + b, 0) / nLoc; this.feederRisk = fr.map(x => x / m);
      this.outHours = new Float64Array(nLoc);
    }

    buildKirana() {
      const cfg = this.cfg, kp = cfg.kirana, r = this.rng.city, n = cfg.city.n_kirana, A = this.A;
      this.n = n;
      let share = []; for (let i = 0; i < n; i++) share.push(r.lognormal(0, kp.size_sigma));
      const s = share.reduce((a, b) => a + b, 0); this.share = share.map(x => x / s);
      const volPc = this.pc.reduce((acc, x, p) => acc + x * this.vol[p], 0);
      this.cap = new Float64Array(n); this.kBase = new Float64Array(n); this.k = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        this.cap[i] = this.share[i] * this.C[0] * volPc * kp.fridge_cover_mean * r.lognormal(0, kp.fridge_sigma);
        this.kBase[i] = r.uniform(kp.k_base_low, kp.k_base_high) + (this.scen.kirana_returns ? kp.returns_k_bonus : 0);
      }
      this.battery = new Uint8Array(n);
      this.applyBatteries(this.scen.battery_fridge_share, this.scen.battery_targeting, r);
      this.k.set(this.kBase);
      this.stock = new Int32Array(n * P * A);
      this.memory = new Float64Array(n * P);
      this.todayD = new Int32Array(n * P); this.todayS = new Int32Array(n * P); this.todayL = new Int32Array(n * P);
      this.todaySpoiled = new Int32Array(n * P); this.lastWaste = new Int32Array(n * P);
      this.lastLate = new Uint8Array(n); this.lastOut = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let p = 0; p < P; p++) {
        const e = this.share[i] * this.C[0] * this.pc[p];
        this.stock[(i * P + p) * A + 1] = Math.round(e);
        this.memory[i * P + p] = e;
      }
      this.shopWaste = new Float64Array(n); this.shopDemand = new Float64Array(n); this.shopLost = new Float64Array(n);
    }

    applyBatteries(share, targeting, r) {
      const n = this.n;
      const had = this.battery ? Array.from(this.battery) : null;
      this.battery = new Uint8Array(n);
      const m = Math.round(share * n);
      if (m > 0) {
        let order = Array.from({ length: n }, (_, i) => i);
        if (targeting === "risk") {
          const tie = order.map(() => r.random());
          order.sort((a, b) => (this.feederRisk[this.loc[b]] - this.feederRisk[this.loc[a]]) || (tie[a] - tie[b]));
        } else order = r.permutation(n);
        for (let j = 0; j < m; j++) this.battery[order[j]] = 1;
      }
      for (let i = 0; i < n; i++) {
        const before = had ? had[i] : 0;
        if (this.battery[i] && !before) this.cap[i] *= 1.25;
        if (!this.battery[i] && before) this.cap[i] /= 1.25;
      }
    }

    buildChannels() {
      const cfg = this.cfg, A = this.A;
      // plant
      this.plant = new Int32Array(P * A);
      this.plantCap = [];
      for (let p = 0; p < P; p++) {
        const tot = this.base[0][p] + this.base[1][p] + this.base[2][p];
        this.plant[p * A] = Math.round(tot * 0.5);
        this.plantCap.push(Math.ceil(tot * cfg.supply.plant_capacity_mult));
      }
      // modern trade
      this.mtDC = new Int32Array(P * A); this.mtShelf = new Int32Array(P * A);
      this.mtCap = []; this.mtSmooth = [];
      for (let p = 0; p < P; p++) {
        this.mtShelf[p * A + 1] = Math.round(this.base[MT][p]);
        this.mtCap.push(Math.ceil(this.base[MT][p] * cfg.mt.shelf_cover));
        this.mtSmooth.push(this.base[MT][p]);
      }
      // quick-commerce
      const S = cfg.city.n_dark_stores; this.S = S;
      this.qcShare = this.rng.city.dirichlet(8, S);
      this.qcStore = new Int32Array(S * P * A);
      this.qcCap = new Int32Array(S * P); this.qcSmooth = this.base[QC].slice();
      for (let s = 0; s < S; s++) for (let p = 0; p < P; p++) {
        const c = Math.ceil(this.qcShare[s] * this.base[QC][p] * cfg.qc.store_cover);
        this.qcCap[s * P + p] = c; this.qcStore[(s * P + p) * A + 1] = Math.floor(c / 2);
      }
      // dark stores sit in the top-right corner of six sectors
      const qcSectors = [0, 2, 5, 6, 9, 11], gx0 = this.cfg.city.grid[0];
      this.qcPos = Array.from({ length: S }, (_, s) => { const l = qcSectors[s % 6]; return [l % gx0 + 0.9, Math.floor(l / gx0) + 0.13]; });
      // pending orders (C x P) and per-shop orders
      this.shopOrders = new Int32Array(this.n * P);
      for (let i = 0; i < this.n; i++) for (let p = 0; p < P; p++) this.shopOrders[i * P + p] = Math.round(this.share[i] * this.base[KIR][p]);
      const oK = [0, 0, 0, 0]; for (let i = 0; i < this.n; i++) for (let p = 0; p < P; p++) oK[p] += this.shopOrders[i * P + p];
      this.pending = [oK, this.base[MT].map(Math.round), this.base[QC].map(Math.round)];
    }

    // ---- regimes
    weekIn(w, lo, hi) { return w >= lo && w <= hi; }
    stepRegimes(week) {
      const p = this.cfg.regimes, r = this.rng.regimes, u = [r.random(), r.random(), r.random()];
      const [lo, hi] = p.summer_weeks;
      let ph;
      if (this.regime.weather === 1) ph = this.weekIn(week, lo, hi + 2) ? p.heat_persist : 0.25;
      else ph = this.weekIn(week, lo, hi) ? p.heat_onset_summer : p.heat_onset_other;
      const ps = this.regime.supply === 1 ? p.stress_persist : p.stress_onset;
      const inWin = p.festival_windows.some(([a, b]) => this.weekIn(week, a, b));
      const pf = !inWin ? 0 : (this.regime.festival === 1 ? p.festival_persist_in_window : p.festival_onset_in_window);
      const forced = this.scen.forced_weather[week];
      this.regime.weather = forced !== undefined ? forced : (u[0] < ph ? 1 : 0);
      this.regime.supply = u[1] < ps ? 1 : 0;
      this.regime.festival = u[2] < pf ? 1 : 0;
    }
    isFlush(w) { const [a, , , d] = this.cfg.regimes.flush_weeks; return w >= a || w <= d; }
    isLean(w) { const [lo, hi] = this.cfg.regimes.summer_weeks; return w >= lo && w <= hi; }
    demandMult() {
      const pr = this.cfg.products, m = [1, 1, 1, 1];
      for (let p = 0; p < P; p++) {
        if (this.regime.weather === 1) m[p] *= pr.heat_mult[p];
        if (this.regime.festival === 1) m[p] *= pr.festival_mult[p];
      }
      return m;
    }
    supplyMult(w) {
      const s = this.cfg.supply; let m = 1;
      if (this.isFlush(w)) m *= s.flush_mult; else if (this.isLean(w)) m *= s.lean_mult;
      if (this.regime.weather === 1) m *= s.heat_mult;
      if (this.regime.supply === 1) m *= s.stressed_mult;
      return m;
    }

    // ---- policy (weekly knobs)
    decide() {
      const avg = [0, 0, 0, 0], H = this.orderHist.slice(-7);
      for (const o of H) for (let p = 0; p < P; p++) avg[p] += o[p] / H.length;
      const heat = this.regime.weather === 1;
      let d;
      if (this.policyName === "Learned" && root.DAIRY_POLICY) {
        const a = this.learnedAction();
        const now = this.demandMult();
        const fc = a.weather_forecast ? avg.map((x, p) => x * now[p] / this.lastMult[p]) : avg;
        d = { plan: fc.map(x => x * (1 + a.margin)), priority: a.kiranas_first ? [KIR, QC, MT] : null,
              twice: a.twice_daily, milkPriority: a.kiranas_first ? [MILK, CURD, PANEER, BUTTER] : null, action: a };
      } else if (this.policyName === "Heat-aware") {
        const now = this.demandMult();
        d = { plan: avg.map((x, p) => x * now[p] / this.lastMult[p] * 1.08),
              priority: heat ? [KIR, QC, MT] : null, twice: heat, milkPriority: [MILK, CURD, PANEER, BUTTER] };
      } else {
        d = { plan: avg.map(x => x * 1.05), priority: null, twice: false, milkPriority: null };
      }
      if (this.scen.force_twice_daily !== null && this.scen.force_twice_daily !== undefined) d.twice = this.scen.force_twice_daily;
      this.decision = d;
    }

    // ---- learned policy: same state encoding and action menu as dairytwin/mdp.py
    learnedState() {
      const w = Math.floor(this.day / 7), L = this.log.slice(-7);
      let d = 0, s = 0; for (const r of L) { d += r.demand[0].reduce((x, y) => x + y, 0); s += r.sales[0].reduce((x, y) => x + y, 0); }
      const fill = L.length ? s / Math.max(d, 1) : 1;
      const season = this.isFlush(w) ? 0 : (this.isLean(w) ? 1 : 2);
      const band = fill < 0.92 ? 0 : (fill < 0.96 ? 1 : 2);
      return ((((this.regime.weather * 2 + this.regime.festival) * 2 + this.regime.supply) * 3 + season) * 3 + band);
    }
    learnedAction() {
      const pol = root.DAIRY_POLICY, s = this.learnedState();
      let a = pol.table[s];
      if (a < 0) a = this.regime.weather === 1 ? pol.default_heat : pol.default_normal;
      this.lastLearnedState = s;
      return pol.actions[a];
    }

    // ---- one day
    step() {
      const cfg = this.cfg, A = this.A, n = this.n, t = this.day, week = Math.floor(t / 7);
      const W = this.watch, wl = [];
      if (t % 7 === 0) { this.lastMult = this.demandMult(); this.stepRegimes(week); this.decide(); }
      else if (this.scen.force_twice_daily !== null && this.scen.force_twice_daily !== undefined) this.decision.twice = this.scen.force_twice_daily;
      const dec = this.decision, mult = this.demandMult(), heat = this.regime.weather === 1;

      // 2. milk and production (base-stock: up to plan net of plant stock)
      const milk = cfg.supply.mean_milk * this.supplyMult(week) * this.rng.supply.lognormal(0, cfg.supply.noise_sigma);
      const r = cfg.products.milk_per_unit, avail0 = [];
      for (let p = 0; p < P; p++) avail0.push(slotTotal(this.plant, p * A, A));
      const want = dec.plan.map((x, p) => Math.min(Math.max(0, Math.ceil(x - avail0[p])), this.plantCap[p]));
      let need = 0; for (let p = 0; p < P; p++) need += want[p] * r[p];
      let made = [0, 0, 0, 0];
      if (need <= milk) made = want.slice();
      else if (!dec.milkPriority) made = want.map(x => x * milk / need);
      else { let left = milk; for (const p of dec.milkPriority) { const tk = Math.max(0, Math.min(want[p], left / r[p])); made[p] = tk; left -= tk * r[p]; } }
      made = made.map(x => Math.max(0, Math.floor(x)));
      let milkUsed = 0; for (let p = 0; p < P; p++) { this.plant[p * A] += made[p]; milkUsed += made[p] * r[p]; }

      // 3. ship against last night's orders
      const orders = this.pending, shipped = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      for (let p = 0; p < P; p++) {
        const av = slotTotal(this.plant, p * A, A), tot = orders[0][p] + orders[1][p] + orders[2][p];
        if (tot <= av) { for (let c = 0; c < 3; c++) shipped[c][p] = orders[c][p]; }
        else if (!dec.priority) { for (let c = 0; c < 3; c++) shipped[c][p] = Math.floor(orders[c][p] * av / Math.max(tot, 1)); }
        else { let left = av; for (const c of dec.priority) { const g = Math.min(orders[c][p], left); shipped[c][p] = g; left -= g; } }
      }
      const prof = [0, 1, 2].map(() => new Int32Array(P * A));
      // kiranas and modern trade load now; quick-commerce is dropped later straight from the plant
      for (const c of [KIR, MT]) for (let p = 0; p < P; p++) slotTake(this.plant, p * A, A, shipped[c][p], prof[c], p * A);
      const qcLeft = shipped[QC].slice(); shipped[QC] = [0, 0, 0, 0];

      // 4. kirana channel: ration, routes, morning, late vans, power cut, noon drop, afternoon
      this.todayD.fill(0); this.todayS.fill(0); this.todayL.fill(0); this.todaySpoiled.fill(0);
      const rd = this.rng.distributor, rk = this.rng.kirana, kp = cfg.kirana;
      const alloc = new Int32Array(n * P);
      for (let p = 0; p < P; p++) {
        let tot = 0; for (let i = 0; i < n; i++) tot += this.shopOrders[i * P + p];
        if (!tot) continue;
        if (shipped[KIR][p] >= tot) { for (let i = 0; i < n; i++) alloc[i * P + p] = this.shopOrders[i * P + p]; continue; }
        let given = 0; const frac = [];
        for (let i = 0; i < n; i++) { const e = this.shopOrders[i * P + p] * shipped[KIR][p] / tot; const b = Math.floor(e); alloc[i * P + p] = b; given += b; frac.push([e - b, i]); }
        let rem = shipped[KIR][p] - given;
        frac.sort((x, y) => y[0] - x[0] + (rd.random() - 0.5) * 1e-9);
        for (let j = 0; j < rem; j++) alloc[frac[j][1] * P + p] += 1;
      }
      // split the shipment's age profile across shops, random service order
      const shopProf = new Int32Array(n * P * A), perm = rd.permutation(n);
      for (let p = 0; p < P; p++) {
        const q = prof[KIR].slice(p * A, p * A + A);
        for (const i of perm) slotTake(q, 0, A, alloc[i * P + p], shopProf, (i * P + p) * A);
      }
      const twice = !!dec.twice;
      const drop1 = new Int32Array(n * P * A), drop2 = twice ? new Int32Array(n * P * A) : null;
      for (let j = 0; j < shopProf.length; j++) {
        if (twice) { drop1[j] = Math.floor(shopProf[j] * 0.55); drop2[j] = shopProf[j] - drop1[j]; } else drop1[j] = shopProf[j];
      }
      const pLate = heat ? kp.p_late_heat : kp.p_late_normal, lateRoute = [];
      for (let rt = 0; rt < cfg.city.n_routes; rt++) lateRoute.push(rd.random() < pLate);
      const late = new Uint8Array(n); for (let i = 0; i < n; i++) late[i] = lateRoute[this.route[i]] ? 1 : 0;
      this.lastLate = late;
      const refused = new Int32Array(P * A);
      const deliver = (prf, which, dropNo) => {
        for (let i = 0; i < n; i++) {
          if (which === "ontime" && late[i]) continue;
          if (which === "late" && !late[i]) continue;
          let inVol = 0;
          for (let p = 0; p < P; p++) inVol += slotTotal(prf, (i * P + p) * A, A) * this.vol[p];
          if (inVol <= 0) continue;
          const free = Math.max(0, this.cap[i] - this.shopVolume(i));
          const fr = inVol > free ? free / inVol : 1;
          let got = 0;
          for (let p = 0; p < P; p++) for (let a = 0; a < A; a++) {
            const j = (i * P + p) * A + a, v = prf[j]; if (!v) continue;
            const acc = Math.floor(v * fr); this.stock[j] += acc; refused[p * A + a] += v - acc; got += acc;
          }
          if (i === W) wl.push(dropNo === 2 ? `R1 noon drop: ${got} units in` : `R1 delivery${late[i] ? " (after 6 am)" : ""}: ${got} units in`);
        }
      };
      // R2: van missing at 6 am -> shopkeeper gets more cautious
      for (let i = 0; i < n; i++) if (late[i] && rk.random() < kp.p_k_late) {
        this.k[i] += kp.k_late_step;
        if (i === W) wl.push(`R2 van late && caution up: k ${(this.k[i] - kp.k_late_step).toFixed(2)} → ${this.k[i].toFixed(2)}`);
      }
      deliver(drop1, "ontime", 1);
      const meanK = (i, p) => this.share[i] * this.C[KIR] * this.pc[p] * mult[p];
      const ms = cfg.products.morning_share;
      const spill = [0, 0, 0, 0];
      const customers = (part) => {
        for (let i = 0; i < n; i++) {
          let lostI = 0;
          for (let p = 0; p < P; p++) {
            const d = rk.poisson(meanK(i, p) * part(p)), off = (i * P + p) * A;
            const sold = slotTake(this.stock, off, A, Math.min(d, slotTotal(this.stock, off, A)), null, 0);
            const lost = d - sold;
            this.todayD[i * P + p] += d; this.todayS[i * P + p] += sold; this.todayL[i * P + p] += lost;
            if (lost > 0) { spill[p] += rk.binomial(lost, P_SPILL); lostI += lost; }
          }
          if (i === W && lostI > 0) wl.push(`R4 shelf empty: ${lostI} customers turned away (some try the app)`);
        }
      };
      customers(p => ms[p]);
      deliver(drop1, "late", 1);
      // power grid: afternoon cuts while the fridge holds the morning stock
      const pw = cfg.power, rp = this.rng.power, hours = this.outHours;
      for (let l = 0; l < this.nLoc; l++) {
        const pr = Math.min((heat ? pw.p_outage_heat : pw.p_outage_normal) * this.feederRisk[l], 0.95);
        const hit = rp.random() < pr;
        const [lo, hi] = heat ? pw.hours_heat : pw.hours_normal;
        hours[l] = hit ? rp.uniform(lo, hi) : 0;
      }
      const fo = this.scen.forced_outages[t];
      if (fo) for (const l in fo) hours[+l] = Math.max(hours[+l], fo[l]);
      if (!this.scen.outages_enabled) hours.fill(0);
      const per4 = cfg.products.outage_age_per_4h;
      let shopHours = 0;
      for (let i = 0; i < n; i++) {
        const h = hours[this.loc[i]], un = this.battery[i] ? Math.max(0, h - pw.battery_hours) : h;
        this.lastOut[i] = un; shopHours += un;
        let spoiledI = 0;
        for (let p = 0; p < P; p++) {
          const ef = un / 4 * per4[p], whole = Math.floor(ef);
          const extra = whole + (rk.random() < ef - whole ? 1 : 0);
          if (extra > 0) { const e = slotAge(this.stock, (i * P + p) * A, A, this.shelf[p], extra); this.todaySpoiled[i * P + p] += e; spoiledI += e; }
        }
        if (i === W && h > 0) wl.push(this.battery[i] && un === 0 ? `R10 power cut ${h.toFixed(1)} h: battery fridge rode it out`
          : `R10 power cut ${h.toFixed(1)} h${this.battery[i] ? ` (battery covered ${pw.battery_hours} h)` : ""}: ${spoiledI} units turned sour`);
      }
      if (twice) deliver(drop2, "all", 2);
      customers(p => 1 - ms[p]);
      for (let a = 0; a < A; a++) for (let p = 0; p < P; p++) this.plant[p * A + a] += refused[p * A + a];

      // 5. modern trade
      const rm = this.rng.mt, dM = [0, 0, 0, 0], sM = [0, 0, 0, 0];
      for (let p = 0; p < P; p++) {
        for (let a = 0; a < A; a++) this.mtDC[p * A + a] += prof[MT][p * A + a];
        const moving = new Int32Array(A); let mv = 0;
        for (let a = 0; a < A; a++) { moving[a] = rm.binomial(this.mtDC[p * A + a], cfg.mt.dc_release_p); mv += moving[a]; }
        const room = Math.max(0, this.mtCap[p] - slotTotal(this.mtShelf, p * A, A));
        let go = Math.min(room, mv);
        for (let a = A - 1; a >= 0 && go > 0; a--) { const g = Math.min(moving[a], go); this.mtDC[p * A + a] -= g; this.mtShelf[p * A + a] += g; go -= g; }
        const d = rm.poisson(this.C[MT] * this.pc[p] * mult[p]);
        const s = slotTake(this.mtShelf, p * A, A, Math.min(d, slotTotal(this.mtShelf, p * A, A)), null, 0);
        dM[p] = d; sM[p] = s;
      }

      // 6. quick-commerce
      const rq = this.rng.qc, dQ = [0, 0, 0, 0], sQ = [0, 0, 0, 0], S = this.S;
      const drop = () => {
        for (let p = 0; p < P; p++) {
          const av = Math.min(qcLeft[p], slotTotal(this.plant, p * A, A)); let tot = 0; const room = [];
          for (let s = 0; s < S; s++) { const rr = Math.max(0, this.qcCap[s * P + p] - slotTotal(this.qcStore, (s * P + p) * A, A)); room.push(rr); tot += rr; }
          for (let s = 0; s < S; s++) {
            const g = tot > av ? Math.floor(room[s] * av / Math.max(tot, 1)) : room[s];
            const got = slotTake(this.plant, p * A, A, g, this.qcStore, (s * P + p) * A);
            qcLeft[p] -= got; shipped[QC][p] += got;
          }
        }
      };
      const serveQ = (part, extra) => {
        for (let p = 0; p < P; p++) {
          const ex = extra ? this.multinomial(rq, extra[p], this.qcShare) : null;
          for (let s = 0; s < S; s++) {
            let d = rq.poisson(this.qcShare[s] * this.C[QC] * this.pc[p] * mult[p] * part(p));
            if (ex) d += ex[s];
            const off = (s * P + p) * A;
            dQ[p] += d; sQ[p] += slotTake(this.qcStore, off, A, Math.min(d, slotTotal(this.qcStore, off, A)), null, 0);
          }
        }
      };
      drop(); serveQ(p => ms[p], null); drop(); serveQ(p => 1 - ms[p], spill);

      // 8. evening close
      const wK = [0, 0, 0, 0], wKret = [0, 0, 0, 0], wP = [0, 0, 0, 0], wM = [0, 0, 0, 0], wQ = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        let exI = 0, heldI = 0;
        for (let p = 0; p < P; p++) {
          const off = (i * P + p) * A;
          const e = slotAge(this.stock, off, A, this.shelf[p], 1) + this.todaySpoiled[i * P + p];
          this.lastWaste[i * P + p] = e; wK[p] += e; if (this.scen.kirana_returns) wKret[p] += e;
          exI += e; heldI += this.todayS[i * P + p] + e;
          this.shopWaste[i] += e; this.shopDemand[i] += this.todayD[i * P + p]; this.shopLost[i] += this.todayL[i * P + p];
        }
        if (exI > kp.heavy_waste_share * Math.max(heldI, 1) && rk.random() < kp.p_k_waste) {
          this.k[i] -= kp.k_waste_step;
          if (i === W) wl.push(`R6 heavy waste && caution down: k → ${this.k[i].toFixed(2)}`);
        }
        this.k[i] = this.kBase[i] + (this.k[i] - this.kBase[i]) * (1 - kp.k_decay);
        this.k[i] = Math.min(kp.k_max, Math.max(kp.k_min, this.k[i]));
        if (i === W) { let sold = 0; for (let p = 0; p < P; p++) sold += this.todayS[i * P + p]; wl.push(`R3 sold ${sold} units today, oldest first`); if (exI) wl.push(`R5 close: ${exI} units expired${this.scen.kirana_returns ? " (returned to the brand)" : ""}`); }
      }
      for (let p = 0; p < P; p++) {
        wP[p] = slotAge(this.plant, p * A, A, this.shelf[p], 1);
        wM[p] = slotAge(this.mtDC, p * A, A, this.shelf[p], 1) + slotAge(this.mtShelf, p * A, A, this.shelf[p], 1);
        wQ[p] = 0;
        for (let s = 0; s < S; s++) wQ[p] += slotAge(this.qcStore, (s * P + p) * A, A, this.shelf[p], 1);
      }

      // 9. tomorrow's orders
      const oK = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        let need = 0; const want = [0, 0, 0, 0];
        for (let p = 0; p < P; p++) {
          const j = i * P + p, seen = this.todayS[j] + kp.lost_weight * this.todayL[j];
          this.memory[j] = kp.memory * this.memory[j] + (1 - kp.memory) * seen;
          const f = 0.5 * seen + 0.5 * this.memory[j], cov = kp.cover_days[p];
          const target = Math.ceil(this.k[i] * f * cov + kp.buffer_sd * Math.sqrt(f * cov));
          want[p] = Math.max(0, target - slotTotal(this.stock, j * A, A));
          need += want[p] * this.vol[p];
        }
        let free = Math.max(0, this.cap[i] - this.shopVolume(i));
        if (twice) for (let p = 0; p < P; p++) free += this.todayS[i * P + p] * ms[p] * this.vol[p];
        const sc = need > free ? free / need : 1;
        let tot = 0;
        for (let p = 0; p < P; p++) { const q = Math.floor(want[p] * sc); this.shopOrders[i * P + p] = q; oK[p] += q; tot += q; }
        if (i === W) wl.push(`R7 order for tomorrow: ${tot} units (k = ${this.k[i].toFixed(2)}${sc < 1 ? ", cut to fit the fridge" : ""})`);
      }
      const oM = [0, 0, 0, 0], oQ = [0, 0, 0, 0];
      for (let p = 0; p < P; p++) {
        const seen = sM[p] + 0.8 * (dM[p] - sM[p]);
        this.mtSmooth[p] = cfg.mt.smoothing * seen + (1 - cfg.mt.smoothing) * this.mtSmooth[p];
        const pipe = slotTotal(this.mtDC, p * A, A) + slotTotal(this.mtShelf, p * A, A);
        oM[p] = Math.max(0, Math.ceil(this.mtSmooth[p] * (1 + cfg.mt.order_cover) - pipe));
        this.qcSmooth[p] = cfg.qc.smoothing * dQ[p] + (1 - cfg.qc.smoothing) * this.qcSmooth[p];
        let pq = 0; for (let s = 0; s < S; s++) pq += slotTotal(this.qcStore, (s * P + p) * A, A);
        oQ[p] = Math.max(0, Math.ceil(this.qcSmooth[p] * (1 + cfg.qc.order_cover) - pq));
      }
      this.pending = [oK, oM, oQ];
      this.orderHist.push([0, 1, 2, 3].map(p => oK[p] + oM[p] + oQ[p]));
      if (this.orderHist.length > 14) this.orderHist.shift();

      // 10. customers drift between channels
      const cp = cfg.consumers, rc = this.rng.consumers;
      let dKt = 0, sKt = 0, lKt = 0, lQt = 0;
      for (let j = 0; j < n * P; j++) { dKt += this.todayD[j]; sKt += this.todayS[j]; lKt += this.todayL[j]; }
      for (let p = 0; p < P; p++) lQt += dQ[p] - sQ[p];
      const kToQ = Math.min(rc.binomial(Math.floor(lKt / cp.units_per_visit), cp.p_migrate_k_to_q), this.C[KIR]);
      const qToK = Math.min(rc.binomial(Math.floor(lQt / cp.units_per_visit), cp.p_migrate_q_to_k), this.C[QC]);
      const fillK = sKt / Math.max(dKt, 1);
      let back = fillK >= cp.winback_fill ? rc.binomial(this.migrated, cp.p_winback) : 0;
      back = Math.min(back, this.C[QC] - qToK);
      this.C[KIR] += qToK + back - kToQ; this.C[QC] += kToQ - qToK - back;
      this.migrated = Math.max(0, Math.min(this.migrated + kToQ - back, this.C[QC]));

      // 11. cash
      const mo = cfg.money, price = cfg.products.retail_price;
      let revenue = 0, cost = milkUsed * cfg.supply.milk_cost;
      let refusedP = [0, 0, 0, 0]; for (let p = 0; p < P; p++) for (let a = 0; a < A; a++) refusedP[p] += refused[p * A + a];
      for (let p = 0; p < P; p++) {
        const wp = mo.wholesale_share * price[p];
        revenue += wp * (shipped[KIR][p] - refusedP[p]) - wp * wKret[p] + wp * (sM[p] + sQ[p]);
        cost += made[p] * cfg.products.processing_cost[p] + slotTotal(this.plant, p * A, A) * mo.holding_cost;
      }
      let nBatt = 0; for (let i = 0; i < n; i++) nBatt += this.battery[i];
      cost += cfg.city.n_routes * (twice ? 2 : 1) * mo.trip_cost + mo.mt_trip_cost + this.S * 2 * mo.qc_trip_cost
        + nBatt * (mo.fridge_cost_per_day + mo.battery_cost_per_day);
      this.cash += revenue - cost;

      const dKp = [0, 0, 0, 0], sKp = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) for (let p = 0; p < P; p++) { dKp[p] += this.todayD[i * P + p]; sKp[p] += this.todayS[i * P + p]; }
      let rNeed = 0; for (let p = 0; p < P; p++) rNeed += (dKp[p] + dM[p] + dQ[p]) * r[p];
      let outLoc = 0; for (let l = 0; l < this.nLoc; l++) if (hours[l] > 0) outLoc++;
      this.log.push({
        day: t, week, weather: this.regime.weather, supply: this.regime.supply, festival: this.regime.festival,
        milk, milkUsed, milkNeed: rNeed, made, demand: [dKp, dM, dQ], sales: [sKp, sM, sQ],
        wasteK: wK, wasteP: wP, wasteM: wM, wasteQ: wQ, outLoc, shopHours, late: late.reduce((a, b) => a + b, 0),
        customers: this.C.slice(), migrated: this.migrated, cash: this.cash, twice, kOrders: oK, spill,
      });
      if (W >= 0) this.watchLog = wl;
      this.day += 1;
    }

    multinomial(r, n, probs) {
      const out = new Array(probs.length).fill(0); let left = n, rest = 1;
      for (let s = 0; s < probs.length - 1 && left > 0; s++) {
        const g = r.binomial(left, Math.min(1, probs[s] / rest)); out[s] = g; left -= g; rest -= probs[s];
      }
      out[probs.length - 1] += left; return out;
    }
    shopVolume(i) {
      let v = 0; const A = this.A;
      for (let p = 0; p < P; p++) v += slotTotal(this.stock, (i * P + p) * A, A) * this.vol[p];
      return v;
    }
    shopStockByAge(i) {
      const A = this.A, out = [];
      for (let p = 0; p < P; p++) out.push(Array.from(this.stock.slice((i * P + p) * A, (i * P + p) * A + Math.min(A, this.shelf[p]))));
      return out;
    }
    unitsInSystem() {
      const A = this.A; let t = [0, 0, 0, 0];
      for (let p = 0; p < P; p++) {
        t[p] += slotTotal(this.plant, p * A, A) + slotTotal(this.mtDC, p * A, A) + slotTotal(this.mtShelf, p * A, A);
        for (let i = 0; i < this.n; i++) t[p] += slotTotal(this.stock, (i * P + p) * A, A);
        for (let s = 0; s < this.S; s++) t[p] += slotTotal(this.qcStore, (s * P + p) * A, A);
      }
      return t;
    }
  }

  const api = { Twin, RNG, P, KIR, MT, QC };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DairyEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
