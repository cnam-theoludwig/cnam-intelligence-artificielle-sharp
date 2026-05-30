const video = document.getElementById("video")
const overlay = document.getElementById("overlay")
const status = document.getElementById("status")
const total = document.getElementById("total")
const overlayContext = overlay.getContext("2d")

const capture = document.createElement("canvas")
const captureContext = capture.getContext("2d")

const FRAME_INTERVAL_MILLISECONDS = 33
const JPEG_QUALITY_RATIO = 0.7
const ACCENT_COLOR = getComputedStyle(document.documentElement)
  .getPropertyValue("--accent")
  .trim()
const LABEL_OFFSET_PIXELS = 6

let inFlight = false

/**
 * @typedef {object} Detection
 * @property {number} x_minimum
 * @property {number} y_minimum
 * @property {number} x_maximum
 * @property {number} y_maximum
 * @property {string} class_name
 * @property {number} confidence
 */

/**
 * @typedef {object} PredictionResult
 * @property {Detection[]} detections
 * @property {number} total_fingers
 */

/**
 * Encode the current capture canvas as a JPEG blob.
 * @returns {Promise<Blob>}
 */
const toJpegBlob = () => {
  return new Promise((resolve) => {
    capture.toBlob(resolve, "image/jpeg", JPEG_QUALITY_RATIO)
  })
}

/**
 * Send one frame to the prediction endpoint.
 * @param {Blob} blob
 * @returns {Promise<PredictionResult>}
 */
const sendFrame = async (blob) => {
  const formData = new FormData()
  formData.append("file", blob, "frame.jpg")
  const response = await fetch("/predict", { method: "POST", body: formData })
  if (!response.ok) {
    throw new Error(`Prediction request failed with status ${response.status}`)
  }
  return await response.json()
}

/**
 * Draw detection boxes and the finger total over the video feed.
 * @param {PredictionResult} result
 * @returns {void}
 */
const draw = (result) => {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height)
  overlayContext.lineWidth = 3
  overlayContext.strokeStyle = ACCENT_COLOR
  overlayContext.fillStyle = ACCENT_COLOR
  overlayContext.font = "20px system-ui, sans-serif"

  for (const detection of result.detections) {
    const width = detection.x_maximum - detection.x_minimum
    const height = detection.y_maximum - detection.y_minimum
    overlayContext.strokeRect(detection.x_minimum, detection.y_minimum, width, height)

    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`
    overlayContext.fillText(
      label,
      detection.x_minimum,
      detection.y_minimum - LABEL_OFFSET_PIXELS,
    )
  }

  total.textContent = result.total_fingers
}

/**
 * Capture a frame, run prediction, and render the result.
 * @returns {Promise<void>}
 */
const tick = async () => {
  if (inFlight) {
    return
  }
  inFlight = true
  try {
    captureContext.drawImage(video, 0, 0, capture.width, capture.height)
    const blob = await toJpegBlob()
    draw(await sendFrame(blob))
  } catch (error) {
    status.textContent = `Prediction error: ${error.message}`
  } finally {
    inFlight = false
  }
}

/**
 * Resolve once the video has known dimensions (videoWidth > 0).
 * @returns {Promise<void>}
 */
const waitForVideoDimensions = () => {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    return Promise.resolve()
  }
  return new Promise((resolve) => {
    video.addEventListener("loadedmetadata", () => resolve(), { once: true })
  })
}

try {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true })
  video.srcObject = stream
  await video.play()
  await waitForVideoDimensions()

  capture.width = video.videoWidth
  capture.height = video.videoHeight
  overlay.width = video.videoWidth
  overlay.height = video.videoHeight

  status.textContent = "Camera running."
  setInterval(tick, FRAME_INTERVAL_MILLISECONDS)
} catch (error) {
  status.textContent = `Camera error: ${error.message}`
}
