// The Sorting Hat - what happens on the web page.
// The thinking is done by the server (app.py + hat_brain.py). This file shows things, plays the
// hat's voice (made by the server, see voice.py), and sends your photo and answers.

const HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"];
const HOUSE_INFO = {
  Gryffindor: { animal: "🦁", words: "Brave, daring and bold" },
  Hufflepuff: { animal: "🦡", words: "Loyal, fair and hard-working" },
  Ravenclaw: { animal: "🦅", words: "Wise, witty and curious" },
  Slytherin: { animal: "🐍", words: "Ambitious, cunning and proud" },
};
const LINES = { // what the hat says (the server prepares these in its voice ahead of time)
  welcome: "Ah... a new student! Step forward, and let me take a look at you.",
  lookAtFace: "Hmm... let me see that face...",
  cantRead: "Hmm, I can't read that picture. Try another one, or skip it.",
  faceNotEverything: "But a face doesn't tell me everything. Let me look inside your mind...",
  noPicture: "No picture? Very well. Let me look inside your mind instead...",
  didntCatch: "Hmm, I didn't catch that... Is app.py still running? Answer again when it is.",
  lostThought: "Hmm, I lost my train of thought... Is app.py still running? Press Try again.",
  backAgain: "Ah, there we are...",
  yourChoice: "Are you sure? Well... the hat does take your choice into account.",
  decided: "Right then... I know just where to put you...",
  noWebcam: "I can't see through your webcam... Choose a photo instead.",
  holdStill: "Look straight at me... and hold still!",
  trying: "Hmm... is this how I sound?",
};
const $ = (selector) => document.querySelector(selector);

// ------------------------------------------------------------------ the hat's voice
// Best: the natural voice made by the server (voice.py). Backup: the browser's built-in voices,
// with a British male voice made a bit deeper and slower.
const PREFERRED_VOICES = ["Daniel", "Google UK English Male", "Arthur", "Oliver", "Grandpa", "Rocko", "Reed", "Eddy"];
const voice = { on: true, natural: null, browser: null, pitch: 0.7, rate: 0.88 };
const canSpeak = "speechSynthesis" in window;
let naturalVoices = []; // [{id, name}] from the server, empty if it can't speak
let defaultNatural = null;

function englishVoices() {
  return canSpeak ? speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("en")) : [];
}

function pickDefaultVoice(voices) {
  const british = voices.filter((v) => v.lang.toLowerCase().replace("_", "-").startsWith("en-gb"));
  for (const name of PREFERRED_VOICES) {
    const match = british.find((v) => v.name.startsWith(name)) || voices.find((v) => v.name.startsWith(name));
    if (match) return match;
  }
  return british[0] || voices[0] || null;
}

function setUpVoices() {
  const select = $("#voice-select");
  select.innerHTML = "";
  for (const v of naturalVoices) select.add(new Option(`🎩 ${v.name} (natural)`, `natural:${v.id}`));
  for (const v of englishVoices()) select.add(new Option(`${v.name} (${v.lang})`, `browser:${v.name}`));
  const saved = load("hatVoice");
  const fallback = pickDefaultVoice(englishVoices());
  const values = [...select.options].map((option) => option.value);
  select.value = values.includes(saved) ? saved
    : naturalVoices.length ? `natural:${defaultNatural}` : fallback ? `browser:${fallback.name}` : "";
  useVoice(select.value);
  $(".voice-controls").hidden = !values.length;
}

function useVoice(value) {
  const name = value.slice(value.indexOf(":") + 1);
  voice.natural = value.startsWith("natural:") ? name : null;
  voice.browser = englishVoices().find((v) => value === `browser:${v.name}`) || pickDefaultVoice(englishVoices());
}

function warmUp() {
  // ask the server to prepare the hat's lines in this voice, so they play without waiting
  if (!voice.natural) return;
  const lines = [...Object.values(LINES), ...HOUSES.map((house) => `${house.toUpperCase()}!`)];
  fetch("/api/warm", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ voice: voice.natural, lines }),
  }).catch(() => {});
}

let stopCurrent = null; // stops the natural voice if it is talking
let utterance = null; // keep a reference, or some browsers forget to tell us when it ends

function stopTalking() {
  if (stopCurrent) stopCurrent();
  if (canSpeak) speechSynthesis.cancel();
}

function speak(text, options = {}) {
  stopTalking();
  if (!voice.on) {
    return new Promise((resolve) => setTimeout(resolve, Math.min(3500, 700 + text.length * 25))); // time to read it
  }
  return voice.natural ? speakNaturally(text, options) : speakWithBrowser(text, options);
}

function speakNaturally(text, options) {
  return new Promise((resolve) => {
    const audio = new Audio(`/api/voice?voice=${encodeURIComponent(voice.natural)}&text=${encodeURIComponent(text)}`);
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(safetyNet);
      if (stopCurrent === stop) stopCurrent = null;
      $("#hat").classList.remove("talking");
      resolve();
    };
    const stop = () => { audio.pause(); finish(); };
    const safetyNet = setTimeout(stop, 30000);
    stopCurrent = stop;
    audio.onplaying = () => $("#hat").classList.add("talking");
    audio.onended = finish;
    audio.onerror = () => { // the server can't speak right now: use the browser's voice for this line
      if (done) return;
      done = true;
      clearTimeout(safetyNet);
      if (stopCurrent === stop) stopCurrent = null;
      speakWithBrowser(text, options).then(resolve);
    };
    audio.play().catch(() => audio.onerror());
  });
}

function speakWithBrowser(text, { pitch = voice.pitch, rate = voice.rate } = {}) {
  return new Promise((resolve) => {
    if (!canSpeak) {
      setTimeout(resolve, Math.min(3500, 700 + text.length * 25));
      return;
    }
    utterance = new SpeechSynthesisUtterance(text);
    if (voice.browser) utterance.voice = voice.browser;
    utterance.pitch = pitch;
    utterance.rate = rate;
    const safetyNet = setTimeout(finish, 4000 + text.length * 120); // in case "end" never arrives
    function finish() {
      clearTimeout(safetyNet);
      $("#hat").classList.remove("talking");
      resolve();
    }
    utterance.onstart = () => $("#hat").classList.add("talking");
    utterance.onend = finish;
    utterance.onerror = finish;
    speechSynthesis.speak(utterance);
  });
}

let typing = null;
const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches; // the computer asks for less motion

function announce(text) {
  $("#speech-sr").textContent = text; // screen readers read the whole sentence once
}

function showSpeech(text) {
  announce(text);
  clearInterval(typing);
  const bubble = $("#speech");
  if (calm) {
    bubble.textContent = text;
    return;
  }
  let shown = 0;
  typing = setInterval(() => {
    bubble.textContent = text.slice(0, ++shown);
    if (shown >= text.length) clearInterval(typing);
  }, 22);
}

async function say(text, options) {
  showSpeech(text);
  await speak(text, options);
}

// ------------------------------------------------------------------ steps of the game

const STEPS = ["enter", "start", "thinking", "face", "question", "tie", "result"];
const game = { face: null, asked: [], points: [0, 0, 0, 0], question: null };
let flow = 0; // goes up at every new step, so an older step that is still talking stops quietly
let busy = false; // waiting for the server, so a double click can't answer twice
let retry = null; // what the "Try again" button does after a failed request

function newStep(name) {
  flow += 1;
  stopTalking();
  for (const step of STEPS) $(`#step-${step}`).hidden = step !== name;
  $("#hat").classList.toggle("thinking", name === "thinking");
  $("#retry").hidden = true;
  return flow;
}

function setAnswerButtons(on) {
  $("#answer-yes").disabled = $("#answer-no").disabled = !on;
}

async function offerRetry(me, action) {
  // The server didn't answer: let the student try again instead of freezing.
  if (me !== flow) return;
  retry = action;
  $("#retry").hidden = false;
  await say(LINES.lostThought);
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The hat is confused.");
  return data;
}

async function welcome() {
  stopWebcam();
  Object.assign(game, { face: null, asked: [], points: [0, 0, 0, 0], question: null });
  document.body.className = "";
  $("#sparkles").innerHTML = "";
  newStep("start");
  await say(LINES.welcome);
}

async function readPhoto(dataUrl) {
  stopWebcam();
  const me = newStep("thinking");
  const talking = say(LINES.lookAtFace);
  let reading;
  try {
    reading = await post("/api/face", { image: dataUrl });
  } catch (error) {
    if (me !== flow) return;
    newStep("start");
    await say(LINES.cantRead);
    return;
  }
  await talking; // let the hat finish its sentence
  if (me !== flow) return;
  game.face = HOUSES.map((house) => reading.probs[house]);
  $("#face-image").src = reading.face;
  newStep("face");
  drawBars($("#face-bars"), reading.probs);
  drawLookalikes(reading.lookalikes);
  const best = HOUSES.reduce((a, b) => (reading.probs[a] >= reading.probs[b] ? a : b));
  const names = reading.lookalikes.slice(0, 2).map((person) => person.name).join(" and ");
  const lost = reading.face_found ? "" : "I can't quite find your face, so I looked at the whole picture. ";
  await say(`${lost}I see a hint of ${best}... and you remind me a little of ${names}.`);
}

async function startQuestions() {
  const me = newStep("thinking");
  await say(game.face ? LINES.faceNotEverything : LINES.noPicture);
  if (me === flow) await nextTurn(null);
}

async function nextTurn(answer) {
  flow += 1; // a new step, but keep the current card on screen while the hat thinks
  const me = flow;
  stopTalking();
  busy = true;
  setAnswerButtons(false);
  let reply;
  try {
    reply = await post("/api/turn", { face: game.face, asked: game.asked, points: game.points, answer });
  } catch (error) {
    busy = false;
    if (me !== flow) return;
    if (answer) { // still on the same question: the student can simply answer again
      setAnswerButtons(true);
      await say(LINES.didntCatch);
    } else {
      await offerRetry(me, () => nextTurn(null));
    }
    return;
  }
  busy = false;
  if (me !== flow) return;
  game.asked = reply.asked;
  game.points = reply.points;
  game.question = reply.question || null;
  if (reply.question) {
    $("#question-number").textContent = `Question ${reply.question.number}`;
    const text = $("#question-text");
    text.textContent = reply.question.text;
    text.classList.remove("fade");
    void text.offsetWidth; // restart the fade-in animation
    text.classList.add("fade");
    const shown = newStep("question");
    setAnswerButtons(true);
    if (reply.whisper) {
      await say(reply.whisper);
      if (shown !== flow) return;
    }
    announce(`Question ${reply.question.number}: ${reply.question.text}`);
    await speak(reply.question.text);
  } else if (reply.tie) {
    await askPreference(reply.tie);
  } else {
    await reveal(null);
  }
}

function answer(yes) {
  if ($("#step-question").hidden || !game.question || busy) return;
  nextTurn({ id: game.question.id, yes });
}

async function askPreference(houses) {
  const buttons = $("#tie-buttons");
  buttons.innerHTML = "";
  for (const house of houses) {
    const button = document.createElement("button");
    button.className = `house-button ${house}`;
    button.textContent = `${HOUSE_INFO[house].animal} ${house}`;
    button.onclick = () => reveal(house);
    buttons.append(button);
  }
  const neither = document.createElement("button");
  neither.className = "quiet";
  neither.textContent = "I don't mind";
  neither.onclick = () => reveal(null);
  buttons.append(neither);
  newStep("tie");
  await say(`Difficult. VERY difficult... ${houses[0]}... or ${houses[1]}? Is there a house you would prefer?`);
}

async function reveal(prefer) {
  const me = newStep("thinking");
  if (prefer) await say(LINES.yourChoice);
  if (me !== flow) return;
  let result;
  try {
    result = await post("/api/result", { face: game.face, points: game.points, prefer });
  } catch (error) {
    await offerRetry(me, () => reveal(prefer));
    return;
  }
  if (me !== flow) return;
  await say(LINES.decided);
  if (me !== flow) return;
  const house = result.house;
  document.body.className = `house-${house}`;
  $("#banner").className = house;
  $("#banner-animal").textContent = HOUSE_INFO[house].animal;
  $("#banner-house").textContent = house;
  $("#banner-words").textContent = HOUSE_INFO[house].words;
  newStep("result");
  drawBars($("#final-bars"), result.probs);
  if (!calm) sparkle(house);
  await say(`${house.toUpperCase()}!`, { pitch: 0.95, rate: 0.8 });
}

// ------------------------------------------------------------------ drawing

function drawBars(container, probs) {
  container.innerHTML = "";
  for (const house of [...HOUSES].sort((a, b) => probs[b] - probs[a])) {
    const row = document.createElement("div");
    row.className = `bar ${house}`;
    const name = document.createElement("span");
    name.className = "bar-name";
    name.textContent = `${HOUSE_INFO[house].animal} ${house}`;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${Math.round(probs[house] * 100)}%`;
    row.append(name, track, value);
    container.append(row);
    setTimeout(() => { fill.style.width = `${probs[house] * 100}%`; }, 50); // grows into place
  }
}

function drawLookalikes(people) {
  const box = $("#lookalikes");
  box.innerHTML = "";
  for (const person of people) {
    const card = document.createElement("div");
    card.className = `lookalike ${person.house}`;
    if (person.picture) {
      const img = document.createElement("img");
      img.src = person.picture;
      img.alt = person.name;
      card.append(img);
    }
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = person.name;
    const house = document.createElement("span");
    house.className = "house";
    house.textContent = person.house;
    card.append(name, house);
    if (typeof person.similarity === "number") {
      const alike = document.createElement("span");
      alike.className = "alike";
      alike.textContent = `${Math.round(Math.max(0, person.similarity) * 100)}% alike`;
      card.append(alike);
    }
    box.append(card);
  }
}

function sparkle(house) {
  const box = $("#sparkles");
  box.className = house;
  box.innerHTML = "";
  for (let i = 0; i < 70; i++) {
    const spark = document.createElement("span");
    spark.style.left = `${Math.random() * 100}%`;
    spark.style.background = ["var(--c1)", "var(--c2)", "var(--gold)"][i % 3];
    spark.style.animationDelay = `${Math.random() * 1.8}s`;
    spark.style.animationDuration = `${2.5 + Math.random() * 2.5}s`;
    box.append(spark);
  }
}

function decorate() {
  const sky = $("#sky");
  for (let i = 0; i < 120; i++) {
    const star = document.createElement("span");
    const size = Math.random() < 0.85 ? 1 : 2;
    Object.assign(star.style, {
      left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%`,
      width: `${size}px`, height: `${size}px`, animationDelay: `${Math.random() * 4}s`,
    });
    sky.append(star);
  }
  const candles = $("#candles");
  for (let i = 0; i < 8; i++) { // just outside the 760px-wide content, on both sides
    const candle = document.createElement("div");
    candle.className = "candle";
    const gap = 25 + Math.random() * 90;
    candle.style.left = i % 2 ? `calc(50% + 380px + ${gap}px)` : `calc(50% - 380px - ${gap}px)`;
    candle.style.top = `${8 + i * 10 + Math.random() * 6}%`;
    candle.style.animationDelay = `${-Math.random() * 7}s`;
    candle.innerHTML = '<span class="flame"></span>';
    candles.append(candle);
  }
}

// ------------------------------------------------------------------ webcam and photos

let stream = null;

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
  } catch (error) {
    await say(LINES.noWebcam);
    return;
  }
  $("#video").srcObject = stream;
  $("#webcam").hidden = false;
  $("#photo-buttons").hidden = true;
  await say(LINES.holdStill);
}

function stopWebcam() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  $("#webcam").hidden = true;
  $("#photo-buttons").hidden = false;
}

function takePicture() {
  const video = $("#video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  readPhoto(canvas.toDataURL("image/jpeg", 0.92));
}

function readFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => readPhoto(reader.result);
  reader.readAsDataURL(file);
}

// ------------------------------------------------------------------ remembering settings

function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private window: just don't remember */ }
}

// ------------------------------------------------------------------ start

async function init() {
  decorate();
  voice.on = load("hatVoiceOn") !== "0";
  $("#mute").textContent = voice.on ? "🔊" : "🔇";

  $("#enter").onclick = welcome;
  $("#use-webcam").onclick = startWebcam;
  $("#choose-photo").onclick = () => $("#photo-input").click();
  $("#photo-input").onchange = (event) => { readFile(event.target.files[0]); event.target.value = ""; };
  $("#skip-photo").onclick = startQuestions;
  $("#capture").onclick = takePicture;
  $("#cancel-webcam").onclick = stopWebcam;
  $("#to-questions").onclick = startQuestions;
  $("#answer-yes").onclick = () => answer(true);
  $("#answer-no").onclick = () => answer(false);
  $("#repeat").onclick = (event) => { event.preventDefault(); if (game.question) speak(game.question.text); };
  $("#again").onclick = welcome;
  $("#mute").onclick = () => {
    voice.on = !voice.on;
    store("hatVoiceOn", voice.on ? "1" : "0");
    $("#mute").textContent = voice.on ? "🔊" : "🔇";
    if (!voice.on) stopTalking();
  };
  $("#voice-select").onchange = (event) => {
    store("hatVoice", event.target.value);
    useVoice(event.target.value);
    say(LINES.trying);
    warmUp();
  };
  $("#retry").onclick = () => {
    const action = retry;
    retry = null;
    $("#retry").hidden = true;
    showSpeech(LINES.backAgain);
    if (action) action();
  };
  document.addEventListener("keydown", (event) => {
    // Only a plain Y or N: Cmd+N opens a new window, and typing in the voice menu isn't an answer.
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    if (event.target instanceof Element && event.target.closest("select, input, textarea")) return;
    const key = event.key.toLowerCase();
    if (key === "y") answer(true);
    if (key === "n") answer(false);
  });
  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    if (!$("#step-start").hidden) readFile(event.dataTransfer.files[0]);
  });

  try {
    const info = await (await fetch("/api/info")).json();
    naturalVoices = info.voices || [];
    defaultNatural = info.default_voice;
    if (!info.can_see) {
      $("#use-webcam").hidden = $("#choose-photo").hidden = true;
      $("#photo-hint").textContent = "The hat has no eyes yet (run python train.py), so it will only ask questions.";
    }
  } catch (error) {
    $("#photo-hint").textContent = "Can't reach the hat's server. Start it with: python app.py";
  }
  setUpVoices();
  if (canSpeak) speechSynthesis.onvoiceschanged = setUpVoices; // browsers load their voices late
  warmUp();
  newStep("enter");
  showSpeech("Psst... over here! Wake me up and I'll find your house.");
}

init();
