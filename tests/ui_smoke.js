// 用真实的服务器 payload 把前端的每个界面都渲染一遍，确保不抛异常。
//
// 存在的理由：前端渲染里任何一处 ReferenceError 都会让界面停在上一屏不动，
// 而 Python 那边的测试一个都抓不到。踩过一次（终局画面调了个已删除的函数，
// 玩家永远卡在"结算中"），所以补这道防线。
//
// 用法：node tests/ui_smoke.js <payloads.json>

const fs = require("fs");
const path = require("path");

const file = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const payloads = file.screens;
const apiConfig = file.config;
const expectedEffects = file.card_effects;
const expectedOrigins = file.origins;

// --- 最小 DOM 桩：只实现 app.js 真正用到的那几个方法 ---
function makeEl(id) {
  const el = {
    id,
    _text: "",
    _html: "",
    dataset: {},
    disabled: false,
    value: "",
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {},
    closest: () => null,
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text; },
    set innerHTML(v) {
      if (v === undefined) throw new Error(`${id}.innerHTML 被赋成了 undefined`);
      this._html = String(v);
    },
    get innerHTML() { return this._html; },
  };
  return el;
}
const els = new Map();
const byClass = new Map();
globalThis.document = {
  getElementById: (id) => {
    if (!els.has(id)) els.set(id, makeEl(id));
    return els.get(id);
  },
  querySelectorAll: (sel) => {
    // 规则速查面板是按 class 找的，得给它真的元素，否则这段代码根本不会跑。
    // 必须缓存：每次返回新对象的话，写进去的 innerHTML 就被扔掉了，等于没测。
    if (sel === ".rulesbox") {
      if (!byClass.has(sel)) {
        byClass.set(sel, [makeEl("rulesbox#1"), makeEl("rulesbox#2")]);
      }
      return byClass.get(sel);
    }
    return [];
  },
  querySelector: () => makeEl("?"),
};
globalThis.localStorage = { getItem: () => "", setItem() {} };
globalThis.location = { protocol: "http:", host: "localhost" };
globalThis.WebSocket = function () {};
globalThis.confirm = () => false;
// 走和浏览器一样的 fetch -> then -> 填 DOM 这条路。
// 之前这里返回空对象 + 测试里直接 setConfig 抄近路，结果整条真实路径根本没被测到。
globalThis.fetch = (url) => {
  if (String(url).includes("/api/config")) {
    return Promise.resolve({ json: () => Promise.resolve(apiConfig) });
  }
  return Promise.resolve({ json: () => Promise.resolve({}) });
};
const realSetTimeout = setTimeout;
globalThis.setTimeout = (fn, ms) => realSetTimeout(fn, ms || 0);
globalThis.clearTimeout = () => {};
globalThis.__MERITOCRACY_TEST__ = true;

const errors = [];
const origError = console.error;
console.error = (...a) => errors.push(a.map(String).join(" "));

const src = fs.readFileSync(path.join(__dirname, "..", "static", "app.js"), "utf8");
(0, eval)(src);

let failed = 0;
async function main() {
for (const p of payloads) {
  errors.length = 0;
  try {
    globalThis.__ui.setState(p.public, p.private, p.my_id);
    if (p.picks) globalThis.__ui.setPicks(p.picks);
    globalThis.__ui.render();
    // 配置是异步取的，取到之后还会再画一次——等它落地，才是浏览器里真实的样子
    await new Promise((r) => setTimeout(r, 0));
    if (p.picks) globalThis.__ui.setPicks(p.picks);
    globalThis.__ui.render();
  } catch (err) {
    console.log(`  ❌ ${p.label}: 渲染直接抛出 ${err.message}`);
    failed++;
    continue;
  }
  if (errors.length) {
    console.log(`  ❌ ${p.label}: ${errors.join(" | ")}`);
    failed++;
  } else {
    console.log(`  ✓ ${p.label}`);
  }
}
// 等 fetch 的 promise 链跑完，规则面板才会被填上
await new Promise((r) => setTimeout(r, 0));

// 牌面上的数字必须和引擎算出来的一致。前端为了即时反馈自己算了一遍，
// 两边一旦漂移，玩家看到的就是错的，而 Python 测试完全抓不到。
if (expectedEffects) {
  const mid = payloads.find((p) => p.label === "行动选择");
  const pv = JSON.parse(JSON.stringify(mid.private));
  pv.rank = expectedEffects.rank;
  pv.hand = ["WORK", "CORRUPT", "GRAFT"].map((c) => ({
    card: c,
    value: expectedEffects.value,
  }));
  globalThis.__ui.setState(mid.public, pv, mid.my_id);
  globalThis.__ui.render();
  await new Promise((r) => setTimeout(r, 0));
  globalThis.__ui.render();
  const shown = document.getElementById("hand").innerHTML.replace(/<[^>]*>/g, " ");
  const want = [
    ["埋头工作 政绩", expectedEffects.WORK],
    ["中饱私囊 金钱", expectedEffects.CORRUPT],
    ["以权谋私 钱", expectedEffects.GRAFT_money],
    ["以权谋私 政绩", expectedEffects.GRAFT_merit],
  ];
  const nums = (shown.match(/\+\d+/g) || []).map((x) => Number(x.slice(1)));
  for (const [label, n] of want) {
    if (!nums.includes(n)) {
      console.log(`  ❌ 牌面数字和引擎对不上：${label} 应该是 ${n}，` +
                  `实际渲染出的数字是 [${nums.join(", ")}]`);
      failed++;
    }
  }
  if (!failed) console.log("  ✓ 牌面数字与引擎一致");
}

// 规则速查面板是渲染到 class 上的，容易出现"写进去又被扔掉"这种静默失败
const boxes = document.querySelectorAll(".rulesbox");
for (const box of boxes) {
  if (!box.innerHTML.includes("升职条件")) {
    console.log(`  ❌ 规则速查面板没渲染出来（${box.id}）`);
    failed++;
  }
}
if (boxes.length && failed === 0) console.log("  ✓ 规则速查");

// 出身会改晋升门槛（官二代的政绩打折）。规则表要是用 /api/config 里那份
// 通用的，他会看到"还差 15"而结算只要 10 —— UI 和引擎对不上。
const vip = payloads.find((p) => p.expect_merit_costs);
if (vip) {
  globalThis.__ui.setState(vip.public, vip.private, vip.my_id);
  globalThis.__ui.render();
  const table = [...document.querySelectorAll(".rulesbox")]
    .map((b) => b.innerHTML.replace(/<[^>]*>/g, " "))
    .join(" ");
  for (const n of vip.expect_merit_costs) {
    if (!new RegExp(`(^|\\s)${n}(\\s|$)`).test(table)) {
      console.log(`  ❌ 官二代的规则表里找不到打折后的政绩门槛 ${n}`);
      failed++;
    }
  }
}

// 出身的名字和技能说明只在 config.py 里定义一处，前端不许自己写一份。
// GRAFT 的 /2 vs /4 就是这么漂移的：JS 里抄了个数字，改配置时没人记得改它。
if (expectedOrigins && expectedOrigins.length) {
  const text = [...document.querySelectorAll(".rulesbox")]
    .map((b) => b.innerHTML.replace(/<[^>]*>/g, " "))
    .join(" ");
  for (const o of expectedOrigins) {
    for (const [what, str] of [["名字", o.name], ["技能名", o.skill],
                               ["说明", o.description]]) {
      if (!text.includes(str)) {
        console.log(`  ❌ 规则里缺出身${what}：${str}`);
        failed++;
      }
    }
  }
  // 挑出身那一屏的候选卡也要带全文案
  const draft = payloads.find((p) => p.label === "挑出身");
  if (draft) {
    globalThis.__ui.setState(draft.public, draft.private, draft.my_id);
    globalThis.__ui.render();
    const box = document.getElementById("originChoices").innerHTML
      .replace(/<[^>]*>/g, " ");
    for (const o of draft.private.origin_choices) {
      if (!box.includes(o.name) || !box.includes(o.skill)) {
        console.log(`  ❌ 挑出身界面没写全：${o.name} / ${o.skill}`);
        failed++;
      }
    }
  }
}

console.error = origError;
process.exit(failed ? 1 : 0);
}
main();
