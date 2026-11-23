/**
 * 손글씨 숫자 인식기 웹 애플리케이션
 * - HTML5 Canvas 마우스/터치 입력 지원
 * - MNIST 표준 규격(28x28, 중심 정렬, 20x20 스케일) 전처리
 * - TensorFlow.js LeNet CNN 모델 기반 0~9 숫자 추론 및 정확도 산출
 */

// ==========================================
// 1. 상태 변수 및 DOM 요소 참조
// ==========================================
let model = null;
window.isModelReady = false;
window.predictDigit = predictDigit;
window.clearCanvas = clearCanvas;
let isDrawing = false;
let hasDrawn = false;
let currentLineWidth = 22;
let autoPredictTimer = null;

// DOM 요소
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const canvasGuide = document.getElementById('canvasGuide');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');

const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const autoPredictToggle = document.getElementById('autoPredictToggle');

const emptyResultState = document.getElementById('emptyResultState');
const resultCard = document.getElementById('resultCard');
const recognizedDigit = document.getElementById('recognizedDigit');
const confidenceBadge = document.getElementById('confidenceBadge');
const confidenceValue = document.getElementById('confidenceValue');
const meterBarFill = document.getElementById('meterBarFill');
const meterText = document.getElementById('meterText');

const toggleDetailsBtn = document.getElementById('toggleDetailsBtn');
const detailsContent = document.getElementById('detailsContent');
const probList = document.getElementById('probList');
const toastEl = document.getElementById('toast');

// ==========================================
// 2. AI 모델 로드 (오프라인 내장 모델 우선 -> 로컬 -> CDN 순서)
// ==========================================
function base64ToArrayBuffer(base64) {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

async function loadAIModel() {
  try {
    // 백엔드 초기화 (WebGL 시도 후 실패 시 CPU 안전 모드로 자동 전환)
    try {
      await tf.ready();
    } catch (backendErr) {
      console.warn('WebGL 백엔드 초기화 실패, CPU 모드로 전환합니다:', backendErr);
      await tf.setBackend('cpu');
      await tf.ready();
    }

    // 1순위: model_data.js 에 포함된 메모리 모델 (CORS 제약 없이 file:// 에서도 100% 작동)
    if (window.PRETRAINED_MODEL_DATA) {
      const weightData = base64ToArrayBuffer(window.PRETRAINED_MODEL_DATA.weightsBase64);
      const modelArtifacts = {
        modelTopology: window.PRETRAINED_MODEL_DATA.modelTopology,
        weightSpecs: window.PRETRAINED_MODEL_DATA.weightSpecs,
        weightData: weightData
      };
      model = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));
    } else {
      // 2순위: CDN 또는 로컬 json 파일 시도
      const MODEL_URL = 'https://cdn.jsdelivr.net/gh/trekhleb/machine-learning-experiments@master/demos/public/models/digits_recognition_cnn/model.json';
      model = await tf.loadLayersModel(MODEL_URL);
    }

    // 워밍업 추론 (첫 추론 지연 제거)
    const dummy = tf.zeros([1, 28, 28, 1]);
    const warmup = model.predict(dummy);
    await warmup.data();
    tf.dispose([dummy, warmup]);

    // 로딩 완료 UI 반영
    window.isModelReady = true;
    statusBadge.classList.add('ready');
    statusText.textContent = '준비 완료 (AI 활성화됨)';
    predictBtn.disabled = false;
  } catch (error) {
    console.error('모델 로드 실패:', error);
    statusText.textContent = '모델 연결 재시도 중...';
    
    // CDN 최후 시도
    try {
      const fallbackUrl = 'https://cdn.jsdelivr.net/gh/trekhleb/machine-learning-experiments@master/demos/public/models/digits_recognition_cnn/model.json';
      model = await tf.loadLayersModel(fallbackUrl);
      window.isModelReady = true;
      statusBadge.classList.add('ready');
      statusText.textContent = '준비 완료 (온라인 모델)';
      predictBtn.disabled = false;
    } catch (e2) {
      statusText.textContent = '모델 로드 오류 (새로고침 해주세요)';
    }
  }
}

// ==========================================
// 3. 캔버스 그리기 관련 이벤트 처리
// ==========================================
function getCanvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

function startDrawing(e) {
  if (e.type === 'touchstart') e.preventDefault();
  isDrawing = true;
  hasDrawn = true;
  canvasGuide.classList.add('hidden');

  const { x, y } = getCanvasCoords(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = currentLineWidth;
  ctx.strokeStyle = '#0f172a';

  // 점만 찍을 때를 위해 선 생성
  ctx.lineTo(x, y);
  ctx.stroke();
}

function draw(e) {
  if (!isDrawing) return;
  if (e.type === 'touchmove') e.preventDefault();

  const { x, y } = getCanvasCoords(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function stopDrawing(e) {
  if (!isDrawing) return;
  isDrawing = false;
  ctx.closePath();

  // 자동 인식 옵션이 켜져 있는 경우, 마우스를 뗀 후 300ms 뒤 자동 예측 실행
  if (autoPredictToggle.checked && hasDrawn) {
    clearTimeout(autoPredictTimer);
    autoPredictTimer = setTimeout(() => {
      predictDigit();
    }, 300);
  }
}

// 마우스 이벤트 등록
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
window.addEventListener('mouseup', stopDrawing);

// 모바일/태블릿 터치 이벤트 등록
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
window.addEventListener('touchend', stopDrawing);

// 펜 굵기 전환 버튼
document.querySelectorAll('.pen-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pen-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentLineWidth = parseInt(btn.dataset.size, 10);
  });
});

// ==========================================
// 4. 지우기 및 초기화
// ==========================================
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hasDrawn = false;
  canvasGuide.classList.remove('hidden');

  // 결과 영역 초기화
  resultCard.classList.add('hidden');
  emptyResultState.classList.remove('hidden');
  meterBarFill.style.width = '0%';
  recognizedDigit.textContent = '-';
  confidenceValue.textContent = '0.0%';

  if (previewCtx) {
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
  }
}

clearBtn.addEventListener('click', clearCanvas);

// ==========================================
// 5. 이미지 전처리 (MNIST 표준 규격 변환)
// ==========================================
/**
 * 캔버스에서 그린 픽셀의 최소/최대 좌표(경계 사각형)를 구합니다.
 */
function getBoundingBox(imageData, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const alpha = imageData.data[idx + 3];
      if (alpha > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX === -1) return null; // 빈 캔버스

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  };
}

/**
 * 사용자 그림을 MNIST 규격 (28x28, 검은 배경에 흰색 글씨, 20x20 크기로 중앙 배치)으로 변환합니다.
 */
function preprocessCanvas() {
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const bbox = getBoundingBox(imgData, canvas.width, canvas.height);

  if (!bbox) return null;

  // 1. 임시 오프스크린 28x28 캔버스 생성
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');

  // 검은 배경 채우기
  tempCtx.fillStyle = '#000000';
  tempCtx.fillRect(0, 0, 28, 28);

  // 2. 가로/세로 비율을 유지하면서 20x20 크기 안에 맞추기
  const maxDim = Math.max(bbox.width, bbox.height);
  const scale = 20.0 / maxDim;
  const scaledWidth = bbox.width * scale;
  const scaledHeight = bbox.height * scale;

  // 3. 28x28 중심(14, 14)에 배치
  const dx = (28 - scaledWidth) / 2;
  const dy = (28 - scaledHeight) / 2;

  // 4. 잘라내서 28x28에 그리기 (흰색 잉크로 전환)
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = bbox.width;
  cropCanvas.height = bbox.height;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(canvas, bbox.minX, bbox.minY, bbox.width, bbox.height, 0, 0, bbox.width, bbox.height);

  // 알파 채널을 흰색(#ffffff)으로 변환
  const cropData = cropCtx.getImageData(0, 0, bbox.width, bbox.height);
  for (let i = 0; i < cropData.data.length; i += 4) {
    const alpha = cropData.data[i + 3];
    cropData.data[i] = 255;     // R
    cropData.data[i + 1] = 255; // G
    cropData.data[i + 2] = 255; // B
    cropData.data[i + 3] = alpha; // Alpha 유지
  }
  cropCtx.putImageData(cropData, 0, 0);

  // 28x28 캔버스에 안티에일리어싱을 켜고 합성
  tempCtx.imageSmoothingEnabled = true;
  tempCtx.drawImage(cropCanvas, dx, dy, scaledWidth, scaledHeight);

  // 미리보기 캔버스에 표시
  previewCtx.drawImage(tempCanvas, 0, 0);

  // 5. 픽셀 데이터 추출 및 0.0 ~ 1.0 정규화 Float32Array 생성
  const finalData = tempCtx.getImageData(0, 0, 28, 28);
  const input = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    // R 또는 G 채널 값 기준 0.0 ~ 1.0 정규화 (MNIST는 검은색=0, 흰색=1)
    const val = finalData.data[i * 4];
    input[i] = val / 255.0;
  }

  return input;
}

// ==========================================
// 6. 숫자 인식 추론 및 결과 표시
// ==========================================
async function predictDigit() {
  if (!isModelReady || !model) {
    showToast('AI 모델을 불러오는 중입니다. 잠시만 기다려주세요.');
    return;
  }

  const inputPixels = preprocessCanvas();
  if (!inputPixels) {
    showToast('숫자를 먼저 네모 칸 안에 그려주세요! 😊');
    return;
  }

  try {
    // [1, 28, 28, 1] 텐서 생성
    const inputTensor = tf.tensor4d(inputPixels, [1, 28, 28, 1]);
    const predictions = model.predict(inputTensor);
    const probabilities = await predictions.data();

    // 텐서 메모리 해제
    tf.dispose([inputTensor, predictions]);

    // 최우선 순위 결과 및 정확도 찾기
    let bestIndex = 0;
    let maxConfidence = -1;
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxConfidence) {
        maxConfidence = probabilities[i];
        bestIndex = i;
      }
    }

    displayResults(bestIndex, maxConfidence, probabilities);
  } catch (error) {
    console.error('인식 중 오류 발생:', error);
    showToast('인식 도중 문제가 발생했습니다. 다시 시도해 주세요.');
  }
}

predictBtn.addEventListener('click', predictDigit);

// ==========================================
// 7. 결과 UI 렌더링
// ==========================================
function displayResults(digit, confidence, probabilities) {
  emptyResultState.classList.add('hidden');
  resultCard.classList.remove('hidden');

  // 1. 인식 결과 숫자
  recognizedDigit.textContent = digit;

  // 2. 정확도 (%)
  const percentage = (confidence * 100).toFixed(1);
  confidenceValue.textContent = `${percentage}%`;

  // 3. 확신도 뱃지 및 게이지 바 스타일링
  confidenceBadge.className = 'confidence-badge';
  meterBarFill.style.width = `${Math.max(percentage, 5)}%`;

  if (confidence >= 0.85) {
    meterText.textContent = '매우 높음 (확실함)';
    meterText.style.color = 'var(--success-color)';
  } else if (confidence >= 0.65) {
    confidenceBadge.classList.add('medium');
    meterText.textContent = '보통 (대체로 맞음)';
    meterText.style.color = 'var(--warning-color)';
  } else {
    confidenceBadge.classList.add('low');
    meterText.textContent = '주의 (비슷한 다른 숫자일 수 있음)';
    meterText.style.color = 'var(--danger-color)';
  }

  // 4. 0~9 모든 후보 확률 막대 리스트 렌더링
  probList.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const probPct = (probabilities[i] * 100).toFixed(1);
    const isTop = (i === digit);

    const row = document.createElement('div');
    row.className = `prob-row ${isTop ? 'top' : ''}`;
    row.innerHTML = `
      <span class="prob-digit">${i}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${Math.max(probPct, 2)}%;"></div>
      </div>
      <span class="prob-val">${probPct}%</span>
    `;
    probList.appendChild(row);
  }
}

// 상세 정보 아코디언 토글
toggleDetailsBtn.addEventListener('click', () => {
  const isHidden = detailsContent.classList.toggle('hidden');
  toggleDetailsBtn.classList.toggle('open', !isHidden);
});

// ==========================================
// 8. 사용자 편의 기능 (토스트 메시지, 단축키)
// ==========================================
let toastTimer = null;
function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.classList.add('hidden');
  }, 2500);
}

// 키보드 단축키 지원 (Space: 인식, Delete/Backspace: 지우기)
window.addEventListener('keydown', (e) => {
  // 입력 폼에 포커스되어 있지 않을 때만 동작
  if (e.target.tagName === 'INPUT') return;

  if (e.code === 'Space') {
    e.preventDefault();
    predictDigit();
  } else if (e.code === 'Delete' || e.code === 'Backspace') {
    e.preventDefault();
    clearCanvas();
  }
});

// ==========================================
// 9. 앱 시작 시 모델 로드 트리거
// ==========================================
window.addEventListener('DOMContentLoaded', () => {
  clearCanvas();
  loadAIModel();
});
