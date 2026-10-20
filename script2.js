
"use strict";

// ========================================
// 1. Canvas and game settings
// ========================================
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

const W = canvas.width;
const H = canvas.height;

const BALL_R = 7;
const PADDLE_W = 76;
const PADDLE_H = 10;
const BRICK_COLS = 5;
const BRICK_ROWS = 3;
const BRICK_W = 72;
const BRICK_H = 16;
const BRICK_GAP = 8;

const ACTIONS = 3; // 0 = left, 1 = stay, 2 = right
const STATE_SIZE = 7;

const GAMMA = 0.95;
const LEARNING_RATE = 0.001;
const BATCH_SIZE = 32;
const REPLAY_LIMIT = 5000;
const WARMUP = 256;
const TRAIN_EVERY = 4;
const TARGET_UPDATE_EVERY = 200;

let onlineModel;
let targetModel;
let replay = [];
let stepCount = 0;
let episode = 0;
let epsilon = 1.0;
let running = true;
let training = false;

let score = 0;
let lives = 3;
let ballX, ballY, dx, dy, paddleX;
let bricks = [];
let previousPaddleContact = false;

let episodeRewards = [];
let episodeReward = 0;
let lastEpisodeScore = 0;
let bestScore = 0;

const statusEl = document.getElementById("status");
const toggleButton = document.getElementById("toggleButton");
const saveButton = document.getElementById("saveButton");
const loadButton = document.getElementById("loadButton");

function createBricks() {
  bricks = [];

  for (let r = 0; r < BRICK_ROWS; r++) {
    for (let c = 0; c < BRICK_COLS; c++) {
      bricks.push({
        x: 20 + c * (BRICK_W + BRICK_GAP),
        y: 35 + r * (BRICK_H + BRICK_GAP),
        alive: true
      });
    }
  }
}

function resetEpisode() {
  ballX = W / 2;
  ballY = H - 45;
  dx = Math.random() < 0.5 ? -2 : 2;
  dy = -2.5;
  paddleX = (W - PADDLE_W) / 2;

  score = 0;
  lives = 3;
  episodeReward = 0;
  previousPaddleContact = false;
  createBricks();
}

function getState() {
  // すべての入力をおおむね -1～1 の範囲に正規化
  return [
    (ballX / W) * 2 - 1,
    (ballY / H) * 2 - 1,
    dx / 4,
    dy / 4,
    ((paddleX + PADDLE_W / 2) / W) * 2 - 1,
    (bricks.filter(b => b.alive).length / (BRICK_ROWS * BRICK_COLS)) * 2 - 1,
    (lives / 3) * 2 - 1
  ];
}

// ========================================
// 2. Deep Q-Network
// ========================================
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [STATE_SIZE],
    units: 64,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: 64,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: ACTIONS,
    activation: "linear"
  }));

  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "meanSquaredError"
  });

  return model;
}

function chooseAction(state) {
  // epsilon-greedy:
  // εの確率で探索、それ以外はQ値が最大の行動を選ぶ
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * ACTIONS);
  }

  return tf.tidy(() => {
    const input = tf.tensor2d([state], [1, STATE_SIZE]);
    const qValues = onlineModel.predict(input);
    return qValues.argMax(1).dataSync()[0];
  });
}

// ========================================
// 3. Experience Replay
// ========================================
function remember(state, action, reward, nextState, done) {
  replay.push({
    state,
    action,
    reward,
    nextState,
    done
  });

  if (replay.length > REPLAY_LIMIT) {
    replay.shift();
  }
}

function sampleBatch() {
  const batch = [];

  for (let i = 0; i < BATCH_SIZE; i++) {
    batch.push(replay[Math.floor(Math.random() * replay.length)]);
  }

  return batch;
}

// ========================================
// 4. DQN training
// ========================================
async function trainStep() {
  if (replay.length < Math.max(BATCH_SIZE, WARMUP)) {
    return;
  }

  const batch = sampleBatch();

  const states = batch.map(item => item.state);
  const nextStates = batch.map(item => item.nextState);

  const currentTensor = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);
  const nextTensor = tf.tensor2d(nextStates, [BATCH_SIZE, STATE_SIZE]);

  let currentQ;
  let nextQ;

  try {
    currentQ = onlineModel.predict(currentTensor);
    nextQ = targetModel.predict(nextTensor);

    const currentValues = await currentQ.array();
    const nextValues = await nextQ.array();

    const targets = currentValues.map((row, i) => {
      const item = batch[i];
      const nextMax = Math.max(...nextValues[i]);

      const updated = row.slice();
      updated[item.action] = item.reward +
        (item.done ? 0 : GAMMA * nextMax);

      return updated;
    });

    const targetTensor = tf.tensor2d(targets, [BATCH_SIZE, ACTIONS]);

    try {
      await onlineModel.fit(currentTensor, targetTensor, {
        epochs: 1,
        batchSize: BATCH_SIZE,
        verbose: 0
      });
    } finally {
      targetTensor.dispose();
    }
  } finally {
    currentTensor.dispose();
    nextTensor.dispose();

    if (currentQ) currentQ.dispose();
    if (nextQ) nextQ.dispose();
  }

  // 定期的にターゲットネットワークを更新
  if (stepCount % TARGET_UPDATE_EVERY === 0) {
    targetModel.setWeights(onlineModel.getWeights());
  }
}

// ========================================
// 5. Game physics and rewards
// ========================================
function moveGame(action) {
  const state = getState();
  let reward = -0.001;
  let done = false;

  // AI action: left, stay, right
  if (action === 0) paddleX -= 5;
  if (action === 2) paddleX += 5;

  paddleX = Math.max(0, Math.min(W - PADDLE_W, paddleX));

  const oldX = ballX;
  const oldY = ballY;

  ballX += dx;
  ballY += dy;

  // Wall collisions
  if (ballX < BALL_R) {
    ballX = BALL_R;
    dx = Math.abs(dx);
  }

  if (ballX > W - BALL_R) {
    ballX = W - BALL_R;
    dx = -Math.abs(dx);
  }

  if (ballY < BALL_R) {
    ballY = BALL_R;
    dy = Math.abs(dy);
  }

  // Paddle collision
  const paddleY = H - 25;

  if (
    dy > 0 &&
    oldY + BALL_R <= paddleY &&
    ballY + BALL_R >= paddleY &&
    ballX >= paddleX &&
    ballX <= paddleX + PADDLE_W
  ) {
    ballY = paddleY - BALL_R;
    dy = -Math.abs(dy);

    // Bounce direction depends on where the ball hits
    const hitPosition =
      (ballX - (paddleX + PADDLE_W / 2)) / (PADDLE_W / 2);

    dx = Math.max(-3.5, Math.min(3.5, hitPosition * 3.5));

    reward += 0.2;
  }

  // Brick collision
  for (const brick of bricks) {
    if (!brick.alive) continue;

    if (
      ballX + BALL_R >= brick.x &&
      ballX - BALL_R <= brick.x + BRICK_W &&
      ballY + BALL_R >= brick.y &&
      ballY - BALL_R <= brick.y + BRICK_H
    ) {
      brick.alive = false;
      dy *= -1;
      score++;
      reward += 1;
      break;
    }
  }

  // Lose a life
  if (ballY - BALL_R > H) {
    lives--;
    reward -= 1;

    if (lives <= 0) {
      done = true;
      reward -= 2;
    } else {
      ballX = W / 2;
      ballY = H - 45;
      dx = Math.random() < 0.5 ? -2 : 2;
      dy = -2.5;
      paddleX = (W - PADDLE_W) / 2;
    }
  }

  // Clear all bricks
  if (bricks.every(b => !b.alive)) {
    reward += 5;
    done = true;
  }

  const nextState = getState();

  remember(state, action, reward, nextState, done);

  episodeReward += reward;
  stepCount++;

  return done;
}

// ========================================
// 6. Rendering
// ========================================
function draw() {
  ctx.clearRect(0, 0, W, H);

  // Bricks
  for (const brick of bricks) {
    if (!brick.alive) continue;

    ctx.fillStyle = "#0095DD";
    ctx.fillRect(brick.x, brick.y, BRICK_W, BRICK_H);
  }

  // Ball
  ctx.beginPath();
  ctx.arc(ballX, ballY, BALL_R, 0, Math.PI * 2);
  ctx.fillStyle = "#E4572E";
  ctx.fill();

  // Paddle
  ctx.fillStyle = "#222";
  ctx.fillRect(paddleX, H - 25, PADDLE_W, PADDLE_H);

  ctx.fillStyle = "#222";
  ctx.font = "14px Arial";
  ctx.fillText("Score: " + score, 8, 18);
  ctx.fillText("Lives: " + lives, W - 75, 18);
}

function updateUI() {
  document.getElementById("episode").textContent = episode;
  document.getElementById("score").textContent = score;

  const average = episodeRewards.length
    ? episodeRewards.reduce((a, b) => a + b, 0) /
      episodeRewards.length
    : 0;

  document.getElementById("reward").textContent =
    average.toFixed(2);

  document.getElementById("epsilon").textContent =
    epsilon.toFixed(3);
}

// ========================================
// 7. Main loop
// ========================================
async function gameLoop() {
  if (!running || training) return;

  training = true;

  try {
    const state = getState();
    const action = chooseAction(state);

    const done = moveGame(action);

    if (stepCount % TRAIN_EVERY === 0) {
      await trainStep();
    }

    // Gradually reduce random exploration
    epsilon = Math.max(0.05, epsilon * 0.9995);

    if (done) {
      episode++;
      lastEpisodeScore = score;
      bestScore = Math.max(bestScore, score);

      episodeRewards.push(episodeReward);
      if (episodeRewards.length > 50) {
        episodeRewards.shift();
      }

      resetEpisode();
    }

    draw();
    updateUI();

    statusEl.textContent =
      "学習中 | Replay: " + replay.length +
      " | Best score: " + bestScore;
  } catch (error) {
    console.error(error);
    statusEl.textContent = "エラー: " + error.message;
    running = false;
  } finally {
    training = false;

    if (running) {
      setTimeout(gameLoop, 20);
    }
  }
}

// ========================================
// 8. Start, pause, save and load
// ========================================
toggleButton.addEventListener("click", () => {
  running = !running;
  toggleButton.textContent = running ? "一時停止" : "再開";

  if (running && !training) {
    gameLoop();
  }
});

saveButton.addEventListener("click", async () => {
  try {
    await onlineModel.save("downloads://breakout-dqn");
    statusEl.textContent = "モデルを保存しました。";
  } catch (error) {
    statusEl.textContent = "保存エラー: " + error.message;
  }
});

loadButton.addEventListener("click", async () => {
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".json";

  fileInput.onchange = async () => {
    if (!fileInput.files.length) return;

    try {
      const modelFile = fileInput.files[0];
      const model = await tf.loadLayersModel(
        tf.io.browserFiles([modelFile])
      );

      if (model.inputs[0].shape[1] !== STATE_SIZE ||
          model.outputs[0].shape[1] !== ACTIONS) {
        model.dispose();
        throw new Error("モデルの入力・出力形状が一致しません。");
      }

      onlineModel.setWeights(model.getWeights());
      targetModel.setWeights(onlineModel.getWeights());
      model.dispose();

      statusEl.textContent =
        "モデルを読み込みました。εを0.05に設定します。";

      epsilon = 0.05;
      updateUI();
    } catch (error) {
      statusEl.textContent = "読み込みエラー: " + error.message;
    }
  };

  fileInput.click();
});

// ========================================
// 9. Initialize
// ========================================
async function start() {
  try {
    await tf.ready();

    onlineModel = createModel();
    targetModel = createModel();
    targetModel.setWeights(onlineModel.getWeights());

    resetEpisode();
    draw();

    statusEl.textContent =
      "準備完了。AIの学習を開始します。";

    toggleButton.disabled = false;
    saveButton.disabled = false;
    loadButton.disabled = false;

    gameLoop();
  } catch (error) {
    console.error(error);
    statusEl.textContent =
      "初期化エラー: " + error.message;
  }
}

start();
