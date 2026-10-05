/* Meritocracy 前端。
 *
 * 客户端只负责显示和提交四种意图：join / select / lock / ready。
 * 所有判定都在服务器。这里拿到的 JSON 里本来就不包含别人的金钱和手牌。
 */

// desc 只是配置还没拉到时的兜底；真正的数字在 renderAction 里按当前官职算。
const CARD_INFO = {
  WORK:          { cn: "埋头工作", desc: "赚政绩（公开）", target: false },
  CORRUPT:       { cn: "中饱私囊", desc: "赚金钱（隐藏，会被举报）", target: false },
  GRAFT:         { cn: "以权谋私", desc: "钱为主，带一点政绩", target: false },
  REPORT:        { cn: "匿名举报", desc: "查他的钱，暗箭", target: true },
  ATTACK:        { cn: "政治攻击", desc: "抢他这轮的功劳归我（明枪）", target: true },
  PROMOTE_MERIT: { cn: "政绩升职", desc: "花政绩升官", target: false },
  PROMOTE_MONEY: { cn: "贿赂升职", desc: "花金钱升官", target: false },
  PROMOTE_ANY:   { cn: "通用升职", desc: "政绩升职，失败就再试贿赂升职", target: false },
  // 红二代「一纸调令」：不在手牌里，每局一次，单独摆在手牌后面
  PROMOTE_FAMILY: { cn: "一纸调令", desc: "家族升职：政绩优先、其次金钱", target: false },
};

// 一纸调令在 local 里的下标（它不在手牌里）
const FAMILY_INDEX = -1;
// 一纸调令不算行动卡、不占出牌位：数"选了几张"时不算它
const actionCount = () => local.filter((x) => x.index !== FAMILY_INDEX).length;

const TOKEN_KEY = "meritocracy.token";
const ROOM_KEY = "meritocracy.room";

let ws = null;
let myId = null;
let myRoom = localStorage.getItem(ROOM_KEY) || "";
let pub = null;
let priv = null;
// 本轮已选的牌：[{index: 手牌下标, card, target}]
let local = [];
let localRound = -1;
let pendingIndex = null; // 正在等你指定目标的那张牌

const $ = (id) => document.getElementById(id);

// 配置里的比例是 "1/2" 这种分数串，直接显示；没拿到配置时给个保守的说法
function fracText(str) {
  return str == null ? "一部分" : String(str);
}

// 买官被举报查实时，行贿的钱打水漂多少（BRIBE_FORFEIT_RATIO）。全部打水漂时说"钱也要不回来"
function bribeLossText() {
  const r = cfgCache ? String(cfgCache.bribe_forfeit_ratio ?? "1") : "1";
  return r === "1" ? "钱也要不回来" : `行贿的钱 ${r} 打水漂`;
}

/* ------------------------------------------------------------------ */
/* WebSocket                                                           */
/* ------------------------------------------------------------------ */

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    $("connState").textContent = "已连接";
    send({ type: "hello", room: myRoom, token: localStorage.getItem(TOKEN_KEY) || "" });
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "identity") {
      myId = msg.player_id;
      localStorage.setItem(TOKEN_KEY, msg.token);
      if (msg.room) {
        myRoom = msg.room;
        localStorage.setItem(ROOM_KEY, msg.room);
      }
    } else if (msg.type === "state") {
      if (msg.room) myRoom = msg.room;
      pub = msg.public;
      priv = msg.private;
      if (priv) myId = priv.player_id;
      render();
    } else if (msg.type === "error") {
      toast(msg.message);
    }
  };

  ws.onclose = () => {
    $("connState").textContent = "断开，重连中…";
    setTimeout(connect, 1200);
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

let toastTimer = null;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------------ */
/* 渲染                                                                */
/* ------------------------------------------------------------------ */

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(id).classList.add("active");
}

function render() {
  // 渲染里任何一处抛异常，都会让 render 提前退出、界面停在上一屏不动
  // （踩过一次：终局画面调了一个已经被删掉的函数，结果永远卡在"结算中"）。
  // 所以这里兜住，至少把屏切过去，并且把错误喊出来。
  try {
    renderInner();
  } catch (err) {
    console.error("渲染出错:", err);
    toast("界面渲染出错，已跳过部分内容");
    try {
      showScreen(SCREEN_OF[pub.phase] || "screen-join");
    } catch (e) {
      /* 实在没办法了 */
    }
  }
}

const SCREEN_OF = {
  LOBBY: "screen-lobby",
  ORIGIN_SELECT: "screen-origin",
  ACTION_SELECTION: "screen-action",
  REVEAL_EVENT: "screen-reveal",
  RESOLUTION: "screen-reveal",
  ROUND_RESULT: "screen-result",
  GAME_OVER: "screen-over",
};

function renderInner() {
  if (!pub) return;

  $("roundInfo").textContent =
    pub.round > 0 ? `第 ${pub.round} / ${pub.max_rounds} 轮` : "未开始";
  $("roomTag").textContent = myRoom ? `房间 ${myRoom}` : "";

  renderLog();
  renderRules();  // 升职条件表里要高亮"我现在这一级"，所以每次渲染都刷一遍

  if (!priv) {
    const full = pub.players.length >= pub.max_players;
    const started = pub.phase !== "LOBBY";
    $("joinHint").textContent = started
      ? `房间 ${myRoom} 已经开局了，你只能旁观。换个房间号或者创建新房间。`
      : full
      ? "这个房间满了（最多 6 人）。"
      : `${pub.min_players}–${pub.max_players} 人，${pub.max_rounds} 轮内爬到国家主席即可获胜。` +
        `每轮发 ${pub.hand_size} 张牌、选 ${pub.picks_per_round} 张打出。`;
    showScreen("screen-join");
    return;
  }

  // 新一轮开始时，把本地选择同步成服务器上的记录（刷新页面也能恢复）
  if (localRound !== pub.round) {
    localRound = pub.round;
    pendingIndex = null;
    local = [];
    const used = new Set();
    (priv.picks || []).forEach((p) => {
      if (p.action === "PROMOTE_FAMILY") {
        local.push({ index: FAMILY_INDEX, card: p.action, target: null });
        return;
      }
      const i = priv.hand.findIndex((d, k) => d.card === p.action && !used.has(k));
      if (i >= 0) used.add(i);
      local.push({ index: i, card: p.action, target: p.target });
    });
  }

  // 先切屏再填内容：这样即使填内容时出错，玩家也不会卡在上一屏
  showScreen(SCREEN_OF[pub.phase] || "screen-join");
  switch (pub.phase) {
    case "LOBBY":            renderLobby();  break;
    case "ORIGIN_SELECT":    renderOrigin(); break;
    case "ACTION_SELECTION": renderAction(); break;
    case "REVEAL_EVENT":
    case "RESOLUTION":       renderReveal(); break;
    case "ROUND_RESULT":     renderResult(); break;
    case "GAME_OVER":        renderOver();   break;
  }
}

function renderLobby() {
  $("roomCode").textContent = myRoom || "----";
  $("lobbyCount").textContent =
    `已加入 ${pub.players.length} 人（${pub.min_players}–${pub.max_players}）`;
  $("lobbyList").innerHTML = pub.players
    .map((p) => {
      const kick =
        priv.is_host && p.is_ai
          ? `<span class="kick" data-kick="${p.id}">移除</span>`
          : "";
      return `<li>${kick}<span class="dot ${p.connected ? "" : "off"}"></span>${esc(p.name)}${
        p.is_ai ? '<span class="aitag">AI</span>' : ""
      }${p.id === pub.host_id ? " <span class=subtle>（房主）</span>" : ""}</li>`;
    })
    .join("");
  const full = pub.players.length >= pub.max_players;
  const canStart = priv.is_host && pub.players.length >= pub.min_players;
  $("addAiBtn").classList.toggle("hidden", !priv.is_host);
  $("addAiBtn").disabled = full;
  $("startBtn").disabled = !canStart;
  $("lobbyHint").textContent = priv.is_host
    ? canStart
      ? "AI 玩家会在服务器上思考，看到的信息和你完全一样。"
      : `还需要至少 ${pub.min_players - pub.players.length} 名玩家（可以加 AI 凑人数）。`
    : `等待房主开始游戏。房间号 ${myRoom}。`;
}

// 结算严格按玩家排的顺序走，所以要把顺序摆明，并且允许调整
function renderOrder() {
  const box = $("orderBox");
  box.classList.toggle("hidden", local.length < 2 || priv.locked);
  if (local.length < 2 || priv.locked) return;

  $("orderList").innerHTML = local
    .map((x, i) => {
      const info = CARD_INFO[x.card] || { cn: x.card };
      const to = x.target !== null ? `<span class="to">→ ${esc(nameOf(x.target))}</span>` : "";
      const step = `<div class="step"><span class="num">${i + 1}</span>${esc(info.cn)}${to}</div>`;
      const arrow = i ? `<span class="arrow">→</span>` : "";
      return arrow + step;
    })
    .join("") + `<button class="swap" id="swapBtn">⇄ 调换顺序</button>`;

  $("swapBtn").onclick = () => {
    // 一纸调令永远最先结算，只调换手牌那几张
    const fam = local.filter((x) => x.index === FAMILY_INDEX);
    local = fam.concat(local.filter((x) => x.index !== FAMILY_INDEX).reverse());
    syncPicks();
    render();
  };

  // 晋升卡摆在哪里，这轮的产出就按哪个官职算——说清楚
  const promoAt = local.findIndex((x) => x.card.startsWith("PROMOTE"));
  // 只有**实际要掏钱**的晋升才会被"刚贪的钱当轮花不出去"拖住。
  // 政绩升职一分钱不用出，排在贪污后面照样当场兑现——别吓唬人。
  const np = priv.next_promotion || {};
  const meritEnough = np.merit_gap === 0;
  const card = promoAt >= 0 ? local[promoAt].card : null;
  // 一纸调令不等举报结算，"刚贪的钱当轮花不出去"对它不适用
  const willSpendMoney =
    card !== null && card !== "PROMOTE_FAMILY" &&
    (card === "PROMOTE_MONEY" ||
      (card === "PROMOTE_ANY" && !meritEnough) ||
      !!np.needs_both);   // 最后一步钱和政绩一起花
  const dirtyBefore =
    promoAt > 0 &&
    willSpendMoney &&
    local.slice(0, promoAt).some((x) => ["CORRUPT", "GRAFT"].includes(x.card));
  const producesAfter = local
    .slice(promoAt + 1)
    .some((x) => ["WORK", "CORRUPT", "GRAFT"].includes(x.card));
  let hint = "排在前面的先结算。";
  if (promoAt >= 0) {
    const previewKey = local[promoAt].card === "PROMOTE_FAMILY" ? "PROMOTE_ANY" : local[promoAt].card;
    const usable = ((priv.card_preview || {})[previewKey] || {}).usable;
    if (usable && promoAt === 0 && producesAfter) {
      hint = "✓ 先升职再干活：这一轮的产出按<b>升职后</b>的官职倍率算，也不会被晋升的 /5 砍掉。";
    } else if (usable && promoAt > 0) {
      hint = "⚠ 晋升卡排在后面：这一轮的产出按<b>现在</b>的官职倍率算，多出来的部分还要被晋升的 /5 砍掉。要不要调换？";
    } else if (!usable && promoAt === 0 && producesAfter) {
      hint = "⚠ 这张晋升卡现在还不够门槛，排在最前面会直接白打。排到后面才能算上这一轮赚到的。";
    } else if (!usable) {
      hint = "这张晋升卡现在还不够门槛，排在后面才能算上这一轮赚到的。";
    }
  }
  if (dirtyBefore) {
    hint +=
      " 另外：晋升卡排在贪污后面，花的是<b>这一轮刚捞的钱</b>，" +
      "要等举报结算完才兑现——万一被举报，这张晋升卡就作废了。";
  }
  $("orderHint").innerHTML = hint;
}

function renderAction() {
  $("myPanel").innerHTML = myPanelHtml();

  const n = pub.picks_per_round;
  const chosen = new Set(local.map((x) => x.index));
  $("handHint").textContent = priv.locked
    ? "已锁定"
    : `选 ${n} 张（已选 ${actionCount()}/${n}）` +
      (pendingIndex !== null ? " —— 请为这张牌指定目标" : "");

  // 每张手牌单独一格（同名牌可以分别选中）
  $("hand").innerHTML = priv.hand
    .map((d, i) => {
      const c = d.card;
      const info = CARD_INFO[c] || { cn: c, desc: "", target: false };
      const order = local.findIndex((x) => x.index === i);
      const pv = (priv.card_preview || {})[c] || {};
      let effect = info.desc;
      let dead = false;
      const mult = Number(priv.card_preview.mult_num) / Number(priv.card_preview.mult_den);
      // 只有开着"大案额外记警告"时，这条提示才是真的。
      // 默认关着（任何一次查实都只记一次警告），不判断的话这行会到处乱报。
      const bigCaseHurts = (cfgCache?.major_corruption_warnings || 1) > 1;
      const line = bigCaseHurts
        ? priv.card_preview.major_corruption_threshold
        : Infinity;
      const scaled = (v) => Math.floor(v * mult);
      // 镜像 rules.graft_merit：先按比例折牌面，再走官职倍率。
      // 这里以前写死了 /2，配置改成 1/4 之后牌面上的政绩就一直是错的。
      const graftMerit = (v) => {
        const [a, b] = String(cfgCache?.graft_merit_ratio ?? "1/4").split("/");
        const r = b ? Number(a) / Number(b) : Number(a);
        return Math.floor(Math.floor(v * r) * mult);
      };
      if (c === "WORK") {
        effect = `政绩 <b>+${scaled(d.value)}</b>` +
          (pv.overtime ? `<span class="subtle">（加班：同一轮打两张，这两张的政绩最后 ×${pv.overtime}，` +
            `再拿 ${pv.overtime} 倍工资的加班费）</span>` : "");
      } else if (c === "CORRUPT") {
        const amt = scaled(d.value);
        effect = `金钱 <b>+${amt}</b>` +
          (amt >= line ? `<span class="warn">⚠这一笔就是大案</span>` : "");
      } else if (c === "GRAFT") {
        const amt = scaled(d.value);
        effect = `钱 <b>+${amt}</b> 政绩 <b>+${graftMerit(d.value)}</b>` +
          (amt >= line ? `<span class="warn">⚠这一笔就是大案</span>` : "");
      } else if (c.startsWith("PROMOTE")) {
        const COUNTER = {
          PROMOTE_MERIT: "花政绩 · 怕政治攻击（会被暂缓）",
          PROMOTE_MONEY: `花金钱 · 怕举报（${bribeLossText()}）`,
          PROMOTE_ANY: "= 政绩升职；政绩不够或被攻击挡下，就再试一次贿赂升职",
        };
        effect =
          (pv.usable ? "✓ 现在可用" : pv.why || "资源不够") +
          `<span class="warn2">${COUNTER[c]}</span>`;
        dead = !pv.usable;
      } else if (c === "REPORT") {
        // 干扰牌以前只写"举报一人"，等于什么都没说
        const cut = cfgCache ? fracText(cfgCache.report_reward_ratio) : "一部分";
        const wmax = cfgCache ? cfgCache.warnings_before_demotion : 2;
        effect =
          `① 举报他<b>贪污受贿</b> → 赃款没收<br>` +
          `② 举报他<b>贿赂升职</b> → 官没了，${bribeLossText()}<br>` +
          `<b>抄到的钱 ${cut} 归我</b>（几个人一起举报就平分这一份` +
          (cfgCache?.report_reward_fee ? `，再扣 ${cfgCache.report_reward_fee} 块跑腿费` : "") + `），` +
          `两样都记一次降职警告（满 ${wmax} 次降一级）<br>` +
          `<span class="warn2">他这轮清白就白打 · 匿名，他不知道是我</span>`;
      } else if (c === "ATTACK") {
        // 关键是说清楚"抢来的政绩归我"——这是这张牌和纯破坏的根本区别，
        // 光写"抢走他的产出"看不出自己能拿到什么。
        const f = cfgCache ? fracText(cfgCache.attack_steal_fraction) : "1/2";
        const fine = (priv.card_preview || {}).idle_penalty;
        effect =
          `① <b>抢功</b>：他埋头工作 → 他这轮挣的政绩 ${f} 归我<br>` +
          `② <b>穿小鞋</b>：他政绩升职 → 放黑料挡住他（他政绩不掉）<br>` +
          `③ <b>戴帽子</b>：他没干正事 → 扣他 ${fine || "一笔"} 政绩` +
          ((priv.card_preview || {}).hat_reward
            ? `，我自己记 ${priv.card_preview.hat_reward} 点功<br>`
            : `（我拿不到）<br>`) +
          `<span class="warn2">明枪：公报点名说是我干的</span>`;
      }
      const cls = [
        "card",
        chosen.has(i) ? "sel" : "",
        priv.locked ? "locked" : "",
        pendingIndex === i ? "pending" : "",
        dead ? "dead" : "",
        c.startsWith("PROMOTE") && pv.usable ? "ready" : "",
      ].join(" ");
      const badge = order >= 0 ? `<div class="count">${order + 1}</div>` : "";
      const tgt =
        order >= 0 && local[order].target !== null
          ? `<div class="tgtname">→ ${esc(nameOf(local[order].target))}</div>`
          : "";
      return `<div class="${cls}" data-index="${i}">${badge}
        <div class="cn">${info.cn}</div>
        <div class="fx">${effect}</div>${tgt}</div>`;
    })
    .join("") + familyCardHtml();

  renderOrder();

  $("targetBox").classList.toggle("hidden", pendingIndex === null);
  if (pendingIndex !== null) {
    $("targets").innerHTML = pub.players
      .filter((p) => p.id !== myId)
      .map(
        (p) =>
          `<div class="tgt" data-target="${p.id}">${esc(p.name)}
            <div class="en">${p.rank_name} · 政绩 ${p.merit}</div></div>`
      )
      .join("");
  }

  // 少打甚至不打都允许，但已选的牌必须都指定好目标
  const ready = local.every((x) => !CARD_INFO[x.card].target || x.target !== null);
  $("lockBtn").disabled = priv.locked || !ready;
  $("lockBtn").textContent = priv.locked
    ? "已锁定，等待其他玩家"
    : local.length === 0
    ? "本轮弃权（会确认）"
    : local.length < n
    ? `只打 ${local.length} 张（会确认）`
    : `锁定这 ${n} 张`;
  $("lockBtn").classList.toggle("warnbtn", !priv.locked && local.length < n);
  const rb = $("redrawBtn");
  const cost = priv.redraw_cost || 0;
  rb.disabled = priv.locked || !priv.redraw_affordable;
  const used = priv.redraws_used_this_round || 0;
  const spent = priv.redraw_spent_this_round || 0;
  const again = used ? `（本轮第 ${used + 1} 次，已花 ${spent}）` : "";
  rb.textContent = !priv.redraw_available
    ? "这一级不能换牌"
    : cost === 0
    ? `免费换一手牌${again}`
    : priv.redraw_affordable
    ? `花 ${cost} 金钱换一手牌${again}`
    : `换一手牌要 ${cost} 金钱，你不够${again}`;

  const waiting = pub.players.filter((p) => !pub.locked_players.includes(p.id));
  $("lockState").textContent = waiting.length
    ? `等待：${waiting.map((p) => p.name).join("、")}`
    : "全部锁定，正在揭示事件…";
  $("forceBtn").classList.toggle("hidden", !(priv.is_host && waiting.length > 0));
  $("abortBtn").classList.toggle("hidden", !priv.is_host);

  $("boardTable").innerHTML = boardHtml(true);
  $("ledgerBox").innerHTML = ledgerHtml();
}

function renderReveal() {
  const e = pub.current_event || {};
  $("revealName").textContent = e.name || "—";
  $("revealDesc").textContent = e.description || "";
}

function renderResult() {
  const r = pub.last_result;
  if (!r) return;
  $("resultEventName").textContent = r.event.name;
  $("resultEventDesc").textContent = r.event.description;

  $("privateResult").innerHTML = privateResultHtml();
  $("publicMessages").innerHTML = r.messages.length
    ? r.messages.map((m) => `<li>${esc(m)}</li>`).join("")
    : "<li class=subtle>本轮风平浪静，无事通报。</li>";
  $("wealthMessages").innerHTML = r.wealth_broadcast.length
    ? r.wealth_broadcast.map((m) => `<li>${esc(m)}</li>`).join("")
    : "<li class=subtle>本轮没有听说谁发财。</li>";
  $("resultBoard").innerHTML = boardHtml(false);

  const iAmReady = pub.ready_players.includes(myId);
  $("readyBtn").disabled = iAmReady;
  $("readyBtn").textContent = iAmReady ? "等待其他玩家…" : "继续";
  const waiting = pub.players.filter(
    (p) => p.connected && !pub.ready_players.includes(p.id)
  );
  $("readyState").textContent = waiting.length
    ? `等待：${waiting.map((p) => p.name).join("、")}`
    : "";
  $("abortBtn2").classList.toggle("hidden", !priv.is_host);
}

function renderOver() {
  $("overReason").textContent = pub.game_over_reason;
  $("postmortem").innerHTML = postmortemHtml();
  $("revealBox").innerHTML = revealHtml();
  $("ledgerBoxOver").innerHTML = ledgerHtml();
  $("finalBoard").innerHTML = boardHtml(false, true);
  $("resetBtn").classList.toggle("hidden", !priv.is_host);
}

/* ------------------------------------------------------------------ */
/* 片段                                                                */
/* ------------------------------------------------------------------ */

function bar(now, need, cls) {
  const pctv = need > 0 ? Math.min(100, Math.round((100 * now) / need)) : 100;
  const done = now >= need;
  return `<div class="track"><div class="fill ${cls}${done ? " done" : ""}"
     style="width:${pctv}%"></div></div>`;
}

// 红二代「一纸调令」：摆在手牌后面的一张特殊牌，每局一次
function familyCardHtml() {
  const fc = priv.family_card;
  if (!fc) return "";
  const order = local.findIndex((x) => x.index === FAMILY_INDEX);
  const enough = ((priv.card_preview || {}).PROMOTE_ANY || {}).usable;
  const status = !fc.usable
    ? esc(fc.why)
    : enough
    ? "✓ 现在够门槛"
    : ((priv.card_preview || {}).PROMOTE_ANY || {}).why || "资源不够，打了会白用";
  const cls = ["card", "family", order >= 0 ? "sel" : "", priv.locked ? "locked" : "",
    !fc.usable || !enough ? "dead" : "", fc.usable && enough ? "ready" : ""].join(" ");
  const badge = order >= 0 ? `<div class="count">${order + 1}</div>` : "";
  return `<div class="${cls}" data-index="${FAMILY_INDEX}">${badge}
    <div class="cn">一纸调令 <span class="subtle">· 红二代每局一次</span></div>
    <div class="fx">${status}<span class="warn2">不占出牌位，用的那一轮最先结算 ·
      政绩优先、其次金钱 · 攻击举报都拦不住 · 不能升主席 · 打出去就算用掉</span></div></div>`;
}

function myPanelHtml() {
  const np = priv.next_promotion;
  if (!np) {
    return `<div class="name">${esc(priv.name)} <span class="rank">· ${priv.rank_name}</span></div>
      <div class="subtle">已经是国家主席了。</div>`;
  }
  const meritOK = np.merit_gap === 0;
  const moneyOK = np.money_gap === 0;
  const both = np.needs_both;
  // 出身会推翻面板上的几句断言。红二代「硬保」降不下来，面板写
  // "再记 1 次就降级"是在骗他；会计「做账」被抄只抄一半。
  const shielded = priv.origin === "RED";
  const launders = priv.origin === "ACCOUNTANT";
  const canGo = both ? meritOK && moneyOK : meritOK || moneyOK;
  const tag = (ok, gap, what) =>
    ok
      ? `<span class="ok">够了 ✓</span>`
      : `<span class="gap">还差 ${gap} ${what}</span>`;
  return `<div class="name">${esc(priv.name)} <span class="rank">· ${priv.rank_name}</span>
      <span class="subtle">→ ${np.to}</span></div>
    <div class="goal">
      <div class="goalrow">
        <span class="lbl">政绩</span>
        <span class="num">${priv.merit} / ${np.merit}</span>
        ${bar(priv.merit, np.merit, "merit")}
        ${tag(meritOK, np.merit_gap, "政绩")}
      </div>
      <div class="goalrow">
        <span class="lbl">金钱</span>
        <span class="num secret">${priv.money} / ${np.money}</span>
        ${bar(priv.money, np.money, "money")}
        ${tag(moneyOK, np.money_gap, "金钱")}
      </div>
      <div class="goalrow">
        <span class="lbl">降职警告</span>
        <span class="num ${priv.warnings ? "warnlbl" : ""}">${priv.warnings} / ${
          pub.warnings_before_demotion
        }</span>
        ${bar(priv.warnings, pub.warnings_before_demotion, "warn")}
        <span class="${priv.warnings && !shielded ? "warnlbl" : "subtle"}">${
          shielded
            ? "上头有人，记满也降不下来"
            : priv.warnings
            ? `再记 ${pub.warnings_before_demotion - priv.warnings} 次就降级`
            : "记录干净"
        }</span>
      </div>
      <div class="goalrow">
        <span class="lbl">工龄</span>
        <span class="num">${priv.tenure} / ${pub.tenure_required}</span>
        ${bar(priv.tenure, pub.tenure_required, "tenure")}
        <span class="${np.tenure_works && np.tenure_gap === 0 ? "ok" : "subtle"}">${
          !np.tenure_works
            ? "熬资历最高只到省级"
            : np.tenure_gap === 0
            ? "本轮自动升 ✓"
            : `再熬 ${np.tenure_gap} 轮自动升`
        }</span>
      </div>
    </div>${
      // 官二代「透风」：选牌阶段就知道本轮全局事件（服务器只发给他自己）
      priv.tipoff_event
        ? `<div class="hintline tipoff">📞 家里来电话：本轮是【${esc(priv.tipoff_event.name)}】` +
          `——${esc(priv.tipoff_event.description)}</div>`
        : ""
    }
    <div class="hintline">官职倍率 <b>x${(priv.card_preview || {}).rank_multiplier || 1}</b>
      · 贪钱或拿钱买官被举报查实 = ${launders ? "赃款没收<b>一半</b>" : "赃款没收"}
      + 记一次降职警告${shielded ? "（但你降不下来）" : ""}<br>${
      meritOK || moneyOK
        ? "<b class=ok>可以升官了 —— 但必须打出对应的晋升卡</b>"
        : "攒够政绩或金钱，再打出晋升卡才能升。工龄满了会自动升。"
    }</div>`;
}

/* 出身 id -> 定义。公开信息，服务器每次都带过来，前端不硬编码任何数值。 */
function originInfo(id) {
  if (!id || !pub.origins) return null;
  return pub.origins.find((o) => o.id === id) || null;
}

function originBadge(id) {
  const o = originInfo(id);
  if (!o) return "";
  return `<span class="origin" title="${esc(o.skill)}：${esc(o.description)}">${esc(
    o.name
  )}</span>`;
}

function renderOrigin() {
  const mine = priv.origin;
  const box = $("originChoices");
  if (mine) {
    const o = originInfo(mine);
    box.innerHTML = `<div class="origincard picked"><div class="oname">${esc(
      o.name
    )}</div><div class="oskill">${esc(o.skill)}</div><div class="odesc">${esc(
      o.description
    )}</div></div>`;
  } else {
    box.innerHTML = (priv.origin_choices || [])
      .map(
        (o) =>
          `<button class="origincard" data-origin="${esc(o.id)}">` +
          `<div class="oname">${esc(o.name)}</div>` +
          `<div class="oskill">${esc(o.skill)}</div>` +
          `<div class="odesc">${esc(o.description)}</div></button>`
      )
      .join("");
  }
  const waiting = (pub.origin_pending || [])
    .map((id) => (pub.players.find((p) => p.id === id) || {}).name)
    .filter(Boolean);
  $("originWait").textContent = waiting.length
    ? `还在等：${waiting.join("、")}`
    : "都选好了，马上开局…";
  $("forceOriginsBtn").classList.toggle("hidden", !priv.is_host || !waiting.length);
}

function boardHtml(showLock, final) {
  const rows = pub.players
    .map((p) => {
      const me = p.id === myId ? " class=me" : "";
      const win = pub.winners.includes(p.id) ? " 🏆" : "";
      const lock = showLock
        ? `<td class="num">${pub.locked_players.includes(p.id) ? '<span class=tick>✓</span>' : "…"}</td>`
        : "";
      const warn = p.warnings
        ? `<span class="warncount" title="降职警告">⚠${p.warnings}</span>`
        : "";
      return `<tr${me}><td><span class="dot ${p.connected ? "" : "off"}"></span>${esc(
        p.name
      )}${win}${warn}${originBadge(p.origin)}</td><td>${p.rank_name}</td><td class="num">${p.merit}</td><td class="num">${
        p.tenure
      }</td>${lock}</tr>`;
    })
    .join("");
  const lockHead = showLock ? "<th>锁定</th>" : "";
  const note = final
    ? '<tr><td colspan="4" class="subtle">金钱是私密数据，终局名次由服务器判定。</td></tr>'
    : "";
  return `<tr><th>玩家</th><th>官职</th><th>政绩</th><th>工龄</th>${lockHead}</tr>${rows}${note}`;
}

function privateResultHtml() {
  const r = priv.private_result;
  if (!r) return '<div class="line subtle">本轮你没有行动。</div>';
  const L = [];
  const names = (r.cards || []).map((c) => (CARD_INFO[c] || { cn: c }).cn);
  L.push(`<div class="line">你打出：<b>${names.join(" + ") || "（未出牌）"}</b></div>`);

  // ---- 金钱流水：一笔一笔写清楚，别让人猜钱去哪了 ----
  const ledger = [];
  if (r.salary) ledger.push([`合法工资`, `+${r.salary}`, "good"]);
  if (r.overtime_pay) ledger.push([`加班费`, `+${r.overtime_pay}`, "good"]);
  // 换牌花的钱以前没列，几笔加起来和"本轮结束"对不上
  if (r.redraw_spent) ledger.push([`重新抽牌`, `−${r.redraw_spent}`, "bad"]);
  if (r.money_gained) ledger.push([`贪污进账`, `+${r.money_gained}`, "good"]);
  if (r.money_from_reports) ledger.push([`分得赃款`, `+${r.money_from_reports}`, "good"]);
  if (r.money_confiscated) ledger.push([`赃款被没收`, `−${r.money_confiscated}`, "bad"]);
  // 做账保住的那部分本来也要被抄。不写出来，玩家根本看不出技能帮了多少
  if (r.laundered) ledger.push([`做账保住的`, `(+${r.laundered})`, "good"]);
  if (r.hush_money_paid) ledger.push([`上下打点压事`, `−${r.hush_money_paid}`, "bad"]);
  if (r.promotion_money_cost) {
    ledger.push([`晋升花掉`, `−${r.promotion_money_cost}`, "bad"]);
  }
  if (ledger.length) {
    L.push('<div class="ledger"><div class="ledhead">金钱</div>');
    ledger.forEach(([k, v, cls]) =>
      L.push(`<div class="ledrow"><span>${k}</span><span class="${cls}">${v}</span></div>`)
    );
    L.push(`<div class="ledrow tot"><span>本轮结束</span><span>${r.money_after}</span></div></div>`);
  }

  // ---- 政绩流水 ----
  const mled = [];
  const overtime = r.overtime_merit || 0;
  if (r.merit_gained - overtime) mled.push([`干活所得`, `+${r.merit_gained - overtime}`, "good"]);
  if (overtime) mled.push([`加班：两张工作翻倍多出来的`, `+${overtime}`, "good"]);
  if (r.merit_from_attacks) mled.push([`攻击所得（抢功 / 记功）`, `+${r.merit_from_attacks}`, "good"]);
  // 被抢走的和被戴帽子扣的是两回事，分开列
  const robbed = r.merit_stolen_by_attackers || 0;
  const fined = (r.attack_merit_loss || 0) - robbed;
  if (robbed) mled.push([`功劳被抢走`, `−${robbed}`, "bad"]);
  if (fined > 0) mled.push([`被戴帽子（没干正事）`, `−${fined}`, "bad"]);
  if (r.merit_wiped_by_attack)
    mled.push([`晋升被拦，政绩作废`, `−${r.merit_wiped_by_attack}`, "bad"]);
  if (r.promotion_merit_cost)
    mled.push([`晋升花掉`, `−${r.promotion_merit_cost}`, "bad"]);
  // 升职后政绩一律打折（÷5），不管是怎么升上去的 —— 贿赂升、熬工龄也一样，
  // 所以不能只在 promotion_merit_cost 非零时才算。用服务端量好的实际打折量。
  if (r.promotion_merit_decay)
    mled.push([`升职后政绩 ÷${pub.merit_overflow_divisor || 5}`,
               `−${r.promotion_merit_decay}`, "bad"]);
  if (mled.length) {
    L.push('<div class="ledger"><div class="ledhead">政绩</div>');
    mled.forEach(([k, v, cls]) =>
      L.push(`<div class="ledrow"><span>${k}</span><span class="${cls}">${v}</span></div>`)
    );
    L.push(`<div class="ledrow tot"><span>本轮结束</span><span>${r.merit_after}</span></div></div>`);
  }

  const PROMO = { MERIT: "政绩晋升", MONEY: "金钱晋升", BOTH: "双条件晋升", TENURE: "工龄晋升" };
  if (r.promotion !== "NONE")
    L.push(`<div class="line good">${PROMO[r.promotion]}成功 → ${esc(priv.rank_name)}</div>`);
  else if (r.promotion_card_played)
    L.push('<div class="line bad">晋升卡没能用上。</div>');
  else if (r.promotion_blocked)
    L.push('<div class="line bad">本轮晋升被阻止。</div>');
  if (r.warnings_issued)
    L.push(
      `<div class="line bad">被举报查实，记降职警告 ${r.warnings_issued} 次` +
        `（累计 ${r.warnings_after} / ${pub.warnings_before_demotion}）。</div>`
    );
  if (r.bribe_lost)
    L.push(`<div class="line bad">行贿的 ${r.bribe_lost} 打了水漂，官也没升成。</div>`);
  if (r.demotion === "MINOR")
    L.push('<div class="line bad">降职警告记满，降一级，警告清空。</div>');
  // 硬保救下来的那一刻是这张牌唯一的高光，不播就白有了
  if (r.origin_shielded_demotion)
    L.push(
      '<div class="line good">降职警告记满了 —— 上头有人打了招呼，' +
        "你的位子纹丝不动。</div>"
    );
  if (r.tenure_reset_by_attack)
    L.push(`<div class="line bad">资历被搅黄，${r.tenure_reset_by_attack} 轮工龄清零</div>`);
  (r.notes || []).forEach((n) => L.push(`<div class="line subtle">${esc(n)}</div>`));
  L.push(
    `<div class="line">当前：${esc(priv.rank_name)} · 金钱 ${r.money_after} · 政绩 ${
      r.merit_after
    } · 工龄 ${r.tenure_after}</div>`
  );
  return L.join("");
}

function postmortemHtml() {
  const pm = priv.postmortem;
  if (!pm) return '<div class="line subtle">没有复盘数据。</div>';
  const out = [];
  const place = pm.is_last ? "垫底" : `第 ${pm.place} / ${pm.total} 名`;
  out.push(`<div class="line"><b>${place}</b></div>`);
  out.push(
    `<div class="line">全场晋升 ${pm.promotions} 次（其他人平均 ${pm.others_avg_promotions} 次）；` +
      `建设回合 ${pm.production_turns} 个（其他人平均 ${pm.others_avg_production} 个）</div>`
  );
  if (!pm.top_causes.length) {
    out.push('<div class="line subtle">各项都不落后，是终局比大小输的。</div>');
  } else {
    out.push('<div class="line">比别人吃亏的地方：</div>');
    pm.top_causes.forEach((c) =>
      out.push(
        `<div class="line bad">· ${esc(c.label)} ${c.mine}（其他人平均 ${c.others_avg}）</div>`
      )
    );
  }
  const st = pm.stats || {};
  out.push(
    `<div class="line subtle">累计：赚政绩 ${st.merit_earned || 0} / 赚钱 ${
      st.money_earned || 0
    } / 抢到政绩 ${st.merit_i_stole || 0} / 分赃 ${st.money_i_took || 0} / 被降级 ${
      st.demoted || 0
    } 次</div>`
  );
  return out.join("");
}

function ledgerHtml() {
  const entries = priv.ledger || [];
  if (!entries.length) return '<div class="subtle">还没有流水。</div>';
  const sign = (v) =>
    `<span class="${v > 0 ? "good" : "bad"}">${v > 0 ? "+" : "−"}${Math.abs(v)}</span>`;
  const out = [
    '<table class="ledtable">',
    "<colgroup><col class=c-rd><col><col class=c-num><col class=c-num></colgroup>",
    '<tr><th>轮</th><th>项目</th><th class="num">金钱</th><th class="num">政绩</th></tr>',
  ];
  entries.forEach((entry) => {
    entry.rows.forEach((r, i) => {
      out.push(
        `<tr><td class="rd">${i === 0 ? entry.round : ""}</td>` +
          `<td>${esc(r.label)}</td>` +
          `<td class="num">${r.money ? sign(r.money) : ""}</td>` +
          `<td class="num">${r.merit ? sign(r.merit) : ""}</td></tr>`
      );
    });
    out.push(
      `<tr class="ledtot${entry.pending ? " ledpending" : ""}">` +
        '<td class="rd"></td>' +
        `<td>${entry.pending ? "本轮进行中（还没结算）" : "余额"}</td>` +
        `<td class="num">${entry.money_after}</td>` +
        `<td class="num">${entry.merit_after}</td></tr>`
    );
  });
  out.push("</table>");
  return out.join("");
}

function revealHtml() {
  const rev = pub.reveal;
  if (!rev) return "";
  const names = rev.names || {};
  const out = ['<table class="board"><tr><th>名次</th><th>玩家</th><th>官职</th>' +
    '<th class="num">金钱</th><th class="num">政绩</th></tr>'];
  rev.standing.forEach((r) => {
    const me = r.player_id === myId ? " class=me" : "";
    out.push(
      `<tr${me}><td>${r.place}${r.won ? " 🏆" : ""}</td>` +
      `<td>${esc(r.name)}${r.is_ai ? '<span class="aitag">AI</span>' : ""}</td>` +
      `<td>${r.rank_name}</td><td class="num">${r.money}</td>` +
      `<td class="num">${r.merit}</td></tr>`
    );
  });
  out.push("</table>");

  out.push('<div class="replay">');
  rev.rounds.forEach((rd) => {
    out.push(`<div class="rdhead">第 ${rd.round} 轮 · ${esc(rd.event)}</div>`);
    rd.players.forEach((p) => {
      const cards = (p.cards || [])
        .map((c, i) => {
          const cn = (CARD_INFO[c] || { cn: c }).cn;
          const t = p.targets[i];
          return t ? `${cn}→${esc(names[t] || t)}` : cn;
        })
        .join("、") || "弃权";
      const tag =
        p.promotion !== "NONE" ? ' <span class="good">升</span>' :
        p.demotion !== "NONE" ? ' <span class="bad">降</span>' : "";
      const me = p.player_id === myId ? " you" : "";
      out.push(
        `<div class="rdrow${me}"><span class="who">${esc(names[p.player_id] || p.player_id)}</span>` +
        `<span class="did">${cards}${tag}</span>` +
        `<span class="num subtle">钱${p.money_after} 绩${p.merit_after}</span></div>`
      );
    });
  });
  out.push("</div>");
  return out.join("");
}

function renderLog() {
  $("logList").innerHTML = (pub.public_messages || [])
    .slice()
    .reverse()
    .map((m) => `<li>${esc(m)}</li>`)
    .join("");
}

let cfgCache = null;
function renderRules() {
  if (!cfgCache) {
    if (renderRules._pending) return;
    renderRules._pending = true;
    fetch("/api/config")
      .then((r) => r.json())
      .then((c) => {
        renderRules._pending = false;
        if (!c || !c.promotion_money_costs) return;
        cfgCache = c;
        // 手牌上的说明（比如攻击抽多少打点费）也要用到配置，
        // 所以配置一到就整屏重画一次，不能只填规则面板。
        if (pub) render();
        else renderRules();
      })
      .catch(() => { renderRules._pending = false; });
    return;
  }
  const html = rulesHtml(cfgCache);
  document.querySelectorAll(".rulesbox").forEach((el) => { el.innerHTML = html; });
}

// 「一图看懂」：试玩反馈说规则太长，先给一张克制关系图 + 升职卡怎么用，细节放后面
function quickRulesHtml(c) {
  const top = c.president_rank - 1;
  const lastMoney = c.promotion_money_costs[top];
  const lastMerit = c.promotion_merit_costs[top];
  const steal = fracText(c.attack_steal_fraction);
  return `
<h4>一图看懂</h4>
<div class="quick">
  <div class="qbox atk">
    <div class="qhead">⚔ 政治攻击 <span class="qsub">明枪 · 公报署名</span></div>
    <div class="qarrow">克 ↓ 政绩路线</div>
    <div class="qitem"><b>埋头工作</b> → 这轮功劳 ${steal} 被抢走</div>
    <div class="qitem"><b>政绩升职</b> → 这轮升不上去（政绩不掉）</div>
    <div class="qitem"><b>没干正事</b>（捞钱/买官/搞人）→ 戴帽子扣政绩${
      c.attack_hat_reward ? "，你记一点功" : ""
    }</div>
  </div>
  <div class="qbox rpt">
    <div class="qhead">🕵 匿名举报 <span class="qsub">暗箭 · 没人知道是谁</span></div>
    <div class="qarrow">克 ↓ 金钱路线</div>
    <div class="qitem"><b>中饱私囊 / 以权谋私</b> → 本轮赃款没收</div>
    <div class="qitem"><b>贿赂升职</b> → 官升不成，${bribeLossText()}</div>
    <div class="qitem">两种都记一次<b>降职警告</b>（满 ${c.warnings_before_demotion} 次降一级）</div>
  </div>
</div>
<p class="rsub">政绩是公开的、钱是暗的。<b>走政绩只怕攻击，走钱只怕举报</b>；
工资谁都碰不到。举报打到这轮清白的人 = 白打；攻击总能落下一样（抢功 / 挡升职 / 戴帽子）。</p>

<table class="rtab">
  <thead><tr><th>升职卡</th><th>花什么</th><th>被攻击</th><th>被举报查实</th></tr></thead>
  <tbody>
    <tr><td><b>政绩升职</b></td><td>政绩</td><td>暂缓，政绩不掉</td><td>不受影响</td></tr>
    <tr><td><b>贿赂升职</b></td><td>钱</td><td>不受影响</td><td>失败，${bribeLossText()}</td></tr>
    <tr><td><b>通用升职</b></td><td colspan="3">= 政绩升职；政绩升职失败（政绩不够、或被攻击挡下）
        就再试一次贿赂升职——走到贿赂那一步就怕举报</td></tr>
    <tr><td><b>一纸调令</b><br><span class="qsub">红二代 · 每局一次</span></td><td>先政绩，不够用钱</td>
        <td>拦不住</td><td>拦不住（用钱那笔记警告）</td></tr>
  </tbody>
</table>
<p class="rsub"><b>升国家主席</b>（${esc(c.rank_names[top])}→${esc(c.rank_names[top + 1])}）：
钱 <b>${lastMoney}</b> 和政绩 <b>${lastMerit}</b> <b>都要够、都要花</b>。
怕谁由你打的卡决定：政绩升职只怕攻击（被拦只是暂缓、一分不亏），贿赂升职只怕举报，
通用升职先按政绩升职算、被攻击挡下就再试贿赂升职，两样都挨就失败。工龄和一纸调令都升不到主席；
同一轮多人登顶只留家底最厚的那个。</p>
`;
}

function rulesHtml(c) {
  const rank = priv ? priv.rank : null;
  const frac = (str) => {
    const [a, b] = String(str).split("/");
    return b ? `${a}/${b}` : a;
  };
  // 倍率写成 1.5 比 3/2 好读
  const mult = (str) => {
    const [a, b] = String(str).split("/");
    const v = b ? Number(a) / Number(b) : Number(a);
    return String(Math.round(v * 100) / 100);
  };

  // 升职条件：把玩家当前所在的那一级高亮，并直接算出"还差多少"。
  // **优先用 priv 里那份**——出身会改门槛（官二代的政绩打折），
  // 用 /api/config 里那份通用的，他看到的数就和结算对不上。
  const moneyCosts = (priv && priv.promotion_money_costs) || c.promotion_money_costs;
  const meritCosts = (priv && priv.promotion_merit_costs) || c.promotion_merit_costs;
  const steps = moneyCosts
    .map((money, i) => {
      const merit = meritCosts[i];
      const both = (c.promotion_requires_both || [])[i];
      const here = rank === i;
      let need = both ? "<b>两样都要</b>" : "二选一";
      if (here && priv) {
        const dm = Math.max(0, money - priv.money);
        const dt = Math.max(0, merit - priv.merit);
        const okm = dm === 0 ? "✓" : `还差 ${dm}`;
        const okt = dt === 0 ? "✓" : `还差 ${dt}`;
        need = both
          ? `<b>两样都要</b> · 钱${okm} 政绩${okt}`
          : `二选一 · 钱${okm} 政绩${okt}`;
      }
      return `<tr class="${here ? "hererow" : ""}">
        <td>${esc(c.rank_names[i])} → ${esc(c.rank_names[i + 1])}</td>
        <td class="num">${money}</td><td class="num">${merit}</td>
        <td>${need}</td></tr>`;
    })
    .join("");

  // 官职待遇
  const perks = c.rank_names
    .map(
      (n, i) =>
        `<tr class="${rank === i ? "hererow" : ""}"><td>${esc(n)}</td>
         <td class="num">×${mult(c.rank_multipliers[i])}</td>
         <td class="num">${c.rank_salary[i]}</td></tr>`
    )
    .join("");

  const events = c.events
    .map((e) => `<li><b>${esc(e.name)}</b>：${esc(e.description)}</li>`)
    .join("");

  const origins = (c.origins || [])
    .map(
      (o) =>
        `<li><b>${esc(o.name)}</b> · <span class="oskill">${esc(o.skill)}</span>：` +
        `${esc(o.description)}</li>`
    )
    .join("");
  const originsBlock = origins
    ? `<h4>出身</h4>
<p class="rsub">开局每人随机发 ${c.origin_choices_offered} 个候选、挑一个，
<b>可能和别人撞</b>。出身整局不变，而且对所有人<b>公开</b>——记分板上写着。</p>
<ul class="rlist">${origins}</ul>`
    : "";

  const rankKeyCn = { money: "金钱", rank: "官职", merit: "政绩" };
  const tiebreak = (c.final_ranking_keys || []).map((k) => rankKeyCn[k] || k).join(" > ");

  return `
${quickRulesHtml(c)}
<p class="rsub">每轮发 ${c.hand_size} 张牌，秘密选 ${c.picks_per_round} 张；所有人锁定后才揭示
全局事件，然后统一结算。最多 ${c.max_rounds} 轮。</p>

<h4>升职条件</h4>
<table class="rtab">
  <thead><tr><th>台阶</th><th class="num">金钱</th><th class="num">政绩</th><th>要求</th></tr></thead>
  <tbody>${steps}</tbody>
</table>
<ul class="rlist">
  <li><b>升职必须打出晋升卡</b>：政绩升职 / 贿赂升职 / 通用升职
      （通用升职 = 政绩升职，失败就再试贿赂升职）。
      红二代另有每局一次的「一纸调令」。</li>
  <li>一轮最多升一级。同一官职连续待满 ${c.tenure_required} 轮自动按工龄升一级
      （工龄升不到主席）。</li>
  <li>升职会<b>花掉</b>门槛那部分资源（贿赂升职花钱，政绩升职花政绩）。</li>
  <li><b>升完一级，剩下的政绩一律 ÷${c.merit_overflow_divisor}</b>——不管你是怎么升上去的，
      贿赂上位、熬工龄熬上去的都一样。金钱${
        c.money_overflow_divisor === 1 ? "没花掉就原样留着" : ` ÷${c.money_overflow_divisor}`
      }。
      所以政绩踩着线升最划算，攒一大堆政绩再花钱升职是白攒。</li>
</ul>

<h4>官职待遇</h4>
<table class="rtab">
  <thead><tr><th>官职</th><th class="num">产出倍率</th><th class="num">每轮工资</th></tr></thead>
  <tbody>${perks}</tbody>
</table>
<p class="rsub"><b>工资在每轮一开始就到账</b>（发牌的同时），所以这笔钱当轮就能花，
比如拿去换一手牌。不占行动位，也不算贪污——举报和反腐风暴都碰不到它。</p>

<h4>出牌顺序（你自己排，严格照着结算）</h4>
<ul class="rlist">
  <li>「晋升卡 → 生产牌」：先升官，这轮的产出按<b>新官职</b>倍率算，也不吃晋升的
      ÷${c.merit_overflow_divisor}。</li>
  <li>「生产牌 → 晋升卡」：先干活，产出按<b>现在</b>的倍率算，多出来的还要被砍。</li>
  <li>门槛还没够就把晋升卡排前面 = <b>白打</b>，系统不会帮你重排。</li>
  <li>晋升卡排在<b>举报/攻击后面</b>，才能花到这轮抄来的赃款和封口费。</li>
  <li><b>换牌要花钱</b>：底价按当前官职定（${c.redraw_costs
        .slice(0, c.president_rank)
        .join(" / ")}），<b>同一轮里每换一次翻 ${c.redraw_cost_growth} 倍</b>
      （${[0, 1, 2].map((i) => c.redraw_costs[0] * c.redraw_cost_growth ** i).join(" → ")}…）。
      基层一轮工资就够换一次，越往上越换不起，而且这笔钱和攒钱升职抢同一个钱包。${
        c.origin_rich_free_redraws
          ? `<br>富二代每轮前 ${c.origin_rich_free_redraws} 次免费，之后从底价开始照常翻倍。`
          : ""
      }</li>
  <li><b>这一轮刚贪来的钱，当轮花不出去。</b>晋升卡排在贪污牌后面的话，
      要等举报结算完、确认没被查实，才会兑现。</li>
  <li><b>被举报查实 = 本轮花钱的晋升作废</b>（贿赂升职、通用升职走钱那条路），
      ${bribeLossText()}；把晋升卡排到贪污前面也躲不掉——官会被撤回来
      （红二代的一纸调令例外：官不撤，只记警告）。
      <b>凭政绩升职不受影响</b>：政绩路线只怕政治攻击。
      （没人举报你，或者你这轮手脚干净，都不受影响。）</li>
</ul>

<h4>行动卡</h4>
<ul class="rlist">
  <li><b>埋头工作</b>：加政绩。政绩是<b>公开</b>的 —— 好处是别人看得见，
      坏处也是别人看得见：本轮产出的 ${frac(c.attack_steal_fraction)} 可能被政治攻击抢走。</li>
  <li><b>中饱私囊</b>：加金钱。钱是<b>隐藏</b>的，别人只能靠坊间传闻猜，
      但这笔钱会被举报和反腐风暴盯上。期望收益是埋头工作的 3 倍。</li>
  <li><b>坊间传闻</b>：每轮结算后点名<b>本轮到手钱最多</b>的人（不报金额）。
      口径 = 工资 + 这轮贪的钱（打点费不扣；<b>被举报/风暴查实的那笔记 0</b>）。
      <b>没人贪了钱还没被抓的那一轮不传</b> —— 所以只要有传闻，就说明这轮至少有人捞了还没被抓，
      但被点名的不一定是他（官大的人光拿工资也可能上榜）。</li>
  <li><b>以权谋私</b>：钱为主（比中饱私囊少），顺带一点政绩 ——
      这点政绩<b>一定少于埋头工作</b>，只是顺手之作。钱同样算贪污。</li>
  <li><b>匿名举报（暗箭）</b>—— 专治走金钱路线的人，两种情况能抓：
      <br>① 他这轮<b>贪污受贿</b> → 本轮赃款全部没收（存款不动）
      <br>② 他这轮<b>贿赂升职</b> → 官升不成，而且<b>${bribeLossText()}</b>
      <br>两种抄到的钱都是<b>${frac(c.report_reward_ratio)} 归你、其余充公</b>
        （几个人一起举报同一个人，就平分这 ${frac(c.report_reward_ratio)}）${
          c.report_reward_fee ? `，每人到手再扣 ${c.report_reward_fee} 块跑腿费` : ""
        }；
        两种都给他记<b>一次降职警告</b>、工龄清零；
      警告攒满 <b>${c.warnings_before_demotion}</b> 次就降一级，然后警告清空重新记。
      <br>他这一轮要是清白的，这张牌就<b>白打</b>。
      <b>匿名</b> —— 他不知道是谁举报的。</li>
  <li><b>政治攻击（明枪）</b>—— 专治走政绩路线的人，一张牌三个作用：
      <br>① <b>抢功</b>：他在埋头工作 → 他这轮挣的政绩
        <b>${frac(c.attack_steal_fraction)} 归你</b>（几个人一起抢就平分这一份，他只掉这么多）
      <br>② <b>穿小鞋</b>：他想靠政绩升职 → 放黑料<b>挡住他</b>，这轮升不上去
        （只是暂缓，他<b>政绩一点不掉</b>）
      <br>③ <b>戴帽子</b>：他这轮<b>没干正事</b>（既没埋头工作、也没凭政绩升职，
        而是在捞钱 / 买官 / 搞别人）→ 给他扣一笔政绩，
        按他的官职算（${c.rank_names.slice(0, c.president_rank).map((n, r) =>
          `${esc(n).replace(/公务员|干部/, "")} ${Math.floor(c.attack_merit_penalty * mult(c.rank_multipliers[r]))}`
        ).join(" / ")}）。
        这笔是<b>罚款，你拿不到</b>${
          c.attack_hat_reward
            ? `；但抓到他不务正业，你自己记一点功（按你的官职：${c.rank_names.slice(0, c.president_rank).map((n, r) =>
                `${esc(n).replace(/公务员|干部/, "")} ${Math.floor(c.attack_hat_reward * mult(c.rank_multipliers[r]))}`
              ).join(" / ")}）`
            : ""
        }
      <br><b>公开署名</b> —— 公报里点名写着是你干的，他下轮知道该找谁算账。
      <br><b>抢官大的人更值</b>：${(() => {
        const top = c.rank_names.length - 2;  // 主席不参与对局
        const at = (r) => Math.floor(c.work_expectation * mult(c.rank_multipliers[r]));
        return `${esc(c.rank_names[top])}一张埋头工作产 ${at(top)} 点，` +
               `${esc(c.rank_names[0])}只产 ${at(0)} 点`;
      })()}。</li>

<h4>升职的克制关系</h4>
<table class="rtab">
  <thead><tr><th>升职方式</th><th>被政治攻击</th><th>被举报查实</th></tr></thead>
  <tbody>
    <tr><td>政绩升职</td><td><b>暂缓</b>（政绩保留）</td><td>不受影响</td></tr>
    <tr><td>贿赂升职</td><td>不受影响</td><td><b>失败</b>，${bribeLossText()}</td></tr>
    <tr><td>通用升职</td><td>政绩升职失败，再试<b>贿赂升职</b></td><td>走政绩那条就不受影响</td></tr>
    <tr><td class="hererow">通用升职 + 两样都挨</td><td colspan="2" class="hererow">
      <b>失败，金钱损失，政绩保留</b></td></tr>
  </tbody>
</table>
<p class="rsub"><b>攻击克政绩路线，举报克金钱路线。</b>通用升职是"有退路"，但退路也会被堵。
<br><b>省级→主席这一步也一样</b>：虽然钱和政绩要一起花，但"你走的是正规程序还是关系"
由你打哪张卡决定 —— 打政绩升职就只怕攻击（被拦下只是暂缓，<b>一分钱不损失</b>），
打贿赂升职就只怕举报。</p>

${originsBlock}

<h4>全局事件（所有人锁定后才揭晓）</h4>
<ul class="rlist">${events}</ul>
<p class="rsub">反腐风暴只查办本轮贪污额排前 ${frac(c.storm_fraction)} 的人（向上取整）。</p>

<h4>怎么算赢</h4>
<ul class="rlist">
  <li>升上<b>${esc(c.rank_names[c.president_rank])}</b>立刻获胜。同一轮多人登顶时比金钱 &gt; 政绩，
      只有家底最厚的当选，其余退回${esc(c.rank_names[c.president_rank - 1])}。</li>
  <li>打满 ${c.max_rounds} 轮还没人登顶，就比 ${tiebreak}。</li>
</ul>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ------------------------------------------------------------------ */
/* 交互                                                                */
/* ------------------------------------------------------------------ */

function nameOrWarn() {
  const name = $("nameInput").value.trim();
  if (!name) {
    toast("请先输入名字");
    return null;
  }
  return name;
}

$("createBtn").onclick = () => {
  const name = nameOrWarn();
  if (name) send({ type: "create", name });
};
$("joinBtn").onclick = () => {
  const name = nameOrWarn();
  if (!name) return;
  const room = $("roomInput").value.trim().toUpperCase();
  if (!room) return toast("请输入房间号");
  send({ type: "join_room", room, name });
};
$("nameInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("createBtn").click();
});
$("roomInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("joinBtn").click();
});

$("redrawBtn").onclick = () => {
  const cost = priv?.redraw_cost || 0;
  // 下一次的价钱由服务器算：富二代免费那次之后是从底价开始，不是 0 x 2
  const next = priv?.redraw_next_cost || 0;
  // 这笔钱和"攒钱升职"抢的是同一个钱包，而且越换越贵——别让人手滑
  if (
    !confirm(
      (cost === 0 ? "免费重新抽一手牌？" : `花 ${cost} 金钱重新抽一手牌？`) +
        "已经选好的牌会清空。\n" +
        `本轮再换下一次要 ${next} 金钱。`
    )
  )
    return;
  send({ type: "redraw" });
};
$("addAiBtn").onclick = () => send({ type: "add_ai" });
$("lobbyList").addEventListener("click", (e) => {
  const el = e.target.closest(".kick");
  if (el) send({ type: "remove_ai", player_id: Number(el.dataset.kick) });
});

$("startBtn").onclick = () => send({ type: "start" });
$("forceBtn").onclick = () => send({ type: "force" });
$("forceOriginsBtn").onclick = () => send({ type: "force_origins" });
// 事件委托挂在容器上（和手牌、大厅列表一个写法）：内容每次重画，
// 监听器只挂一次，不会越积越多
$("originChoices").addEventListener("click", (e) => {
  const card = e.target.closest("[data-origin]");
  if (card) send({ type: "choose_origin", origin: card.dataset.origin });
});
$("readyBtn").onclick = () => send({ type: "ready" });
function askReset() {
  const playing = pub && pub.phase !== "LOBBY" && pub.phase !== "GAME_OVER";
  const msg = playing
    ? "确定结束这一局吗？当前进度会全部清空，所有人（含 AI）回到大厅重开。"
    : "确定要开新的一局吗？当前进度会被清空。";
  if (confirm(msg)) send({ type: "reset" });
}
$("resetBtn").onclick = askReset;
$("abortBtn").onclick = askReset;
$("abortBtn2").onclick = askReset;

function syncPicks() {
  // 还没指定目标的干扰牌先别发：服务器会整批打回来一句"这张卡需要选择一个目标"，
  // 结果就是每次点举报/攻击都先弹一个红字错误。锁定时会把完整的一套再发一遍。
  send({ type: "select", picks: readyPicks() });
}

function readyPicks() {
  return local
    .filter((x) => !CARD_INFO[x.card].target || x.target !== null)
    .map((x) =>
      x.index === FAMILY_INDEX
        ? { action: "PROMOTE_FAMILY", target: null }
        : { index: x.index, target: x.target }
    );
}

$("lockBtn").onclick = () => {
  // 少打牌是合法选择，但不该和手滑长得一样——所以要确认一次
  const n = pub.picks_per_round;
  const picked = actionCount();
  if (picked < n) {
    const wasted = n - picked;
    const msg =
      picked === 0
        ? `本轮弃权？${n} 个行动位都会浪费掉。`
        : `你只选了 ${picked} 张，还有 ${wasted} 个行动位会浪费掉。确定就这么打？`;
    if (!confirm(msg)) return;
  }
  send({ type: "lock", picks: readyPicks() });
};

$("hand").addEventListener("click", (e) => {
  const el = e.target.closest(".card");
  if (!el || priv?.locked) return;
  const i = Number(el.dataset.index);
  const at = local.findIndex((x) => x.index === i);
  if (at >= 0) {
    local.splice(at, 1); // 再点一下 = 取消
    if (pendingIndex === i) pendingIndex = null;
  } else {
    if (i === FAMILY_INDEX) {
      // 不占出牌位，而且永远最先结算
      if (!(priv.family_card || {}).usable) {
        return toast((priv.family_card || {}).why || "现在不能用一纸调令");
      }
      local.unshift({ index: FAMILY_INDEX, card: "PROMOTE_FAMILY", target: null });
      syncPicks();
      render();
      return;
    }
    if (actionCount() >= pub.picks_per_round) return toast(`最多选 ${pub.picks_per_round} 张`);
    const card = priv.hand[i].card;
    local.push({ index: i, card, target: null });
    pendingIndex = CARD_INFO[card].target ? i : null;
  }
  syncPicks();
  render();
});

$("targets").addEventListener("click", (e) => {
  const el = e.target.closest(".tgt");
  if (!el || priv?.locked || pendingIndex === null) return;
  const entry = local.find((x) => x.index === pendingIndex);
  if (entry) entry.target = Number(el.dataset.target);
  pendingIndex = null;
  syncPicks();
  render();
});

function nameOf(id) {
  const p = (pub.players || []).find((x) => x.id === id);
  return p ? p.name : "?";
}

if (typeof globalThis.__MERITOCRACY_TEST__ === "undefined") {
  connect();
} else {
  // 给 tests/ui_smoke.js 用：不连服务器，只把渲染入口暴露出来
  globalThis.__ui = {
    render,
    setState: (p, v, id) => {
      pub = p;
      priv = v;
      myId = id;
      localRound = -1;
      local = [];
    },
    setConfig: (c) => {
      cfgCache = c;
    },
    setPicks: (picks) => {
      local = picks;
      localRound = pub.round;
    },
  };
}
