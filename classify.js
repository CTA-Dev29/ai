const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const path = require('path');

async function run() {
  const modelPath = path.resolve(__dirname, 'models/model.json');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  console.log("✅ Model berhasil dimuat!");

  const imagePath = './uploads/img-1750220471375.jpg';
  const image = await loadImage(imagePath);

  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 224, 224);
  const input = tf.browser.fromPixels(canvas).expandDims(0).toFloat().div(255);

  const prediction = await model.predict(input).data();
  console.log("📊 Prediction:", prediction);
}

run();
