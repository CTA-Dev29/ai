const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const tf = require('@tensorflow/tfjs');
const { createCanvas, loadImage } = require('canvas');
require('dotenv').config();
const cors = require('cors');

const app = express();
const PORT = 5000;
app.use(cors({ origin: 'http://localhost:3000' }));

// Upload setup
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, 'img-' + Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

// Label sesuai urutan pelatihan model Teachable Machine
const labels = ['Plastik', 'Kertas', 'Organik', 'Logam', 'Kaca']; // EDIT sesuai model kamu

// Klasifikasi gambar lokal
async function classifyLocal(imagePath) {
const modelPath = path.resolve(__dirname, 'models/model.json');
const model = await tf.loadLayersModel(`file://${modelPath}`);

  const image = await loadImage(imagePath);
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 224, 224);
  const input = tf.browser.fromPixels(canvas).expandDims(0).toFloat().div(255);

  const prediction = await model.predict(input).data();
  const maxIndex = prediction.indexOf(Math.max(...prediction));

  return {
    label: labels[maxIndex],
    confidence: prediction[maxIndex]
  };
}

// Cari video YouTube via Google CSE
async function cariVideoYoutube(query) {
  try {
    const url = `https://www.googleapis.com/customsearch/v1`;
    const response = await axios.get(url, {
      params: {
        key: process.env.GOOGLE_API_KEY,
        cx: process.env.GOOGLE_CSE_ID,
        q: query + " site:youtube.com",
      },
    });

    const item = response.data.items?.[0];
    return item ? item.link : "Tidak ditemukan";
  } catch (error) {
    console.error('❌ Gagal cari video:', error.response?.data || error.message);
    return "Tidak ditemukan";
  }
}

// API Endpoint utama
app.post('/api/classify', upload.single('image'), async (req, res) => {
  try {
    const imagePath = req.file.path;

    // 1. Klasifikasi lokal
    const result = await classifyLocal(imagePath);
    const jenisSampah = result.label;

    // 2. Tanya ke Groq
    const prompt = `
Jenis sampah: ${jenisSampah}
Tolong bantu saya:
1. Jelaskan cara mendaur ulang sampah ini.
2. Berikan 3 ide kerajinan tangan dari sampah ini.
Jawab dalam Bahasa Indonesia yang santai dan mudah dipahami.
`;

    const groqResponse = await axios.post(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        model: "llama3-70b-8192",
        messages: [
          {
            role: 'system',
            content: 'Kamu adalah asisten ramah yang menjawab dalam bahasa Indonesia dengan gaya santai.'
          },
          {
            role: 'user',
            content: prompt,
          }
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );

    const aiResult = groqResponse.data.choices[0].message.content;

    // 3. Cari video
    const videoLink = await cariVideoYoutube(`cara daur ulang sampah ${jenisSampah}`);

    res.json({
      jenis: jenisSampah,
      confidence: result.confidence,
      penjelasan: aiResult,
      video: `📺 Video referensi: ${videoLink}`
    });

  } catch (err) {
    console.error('❌ ERROR:', err.response?.data || err.message || err);
    res.status(500).json({ error: 'Terjadi kesalahan saat memproses gambar.' });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server berjalan di http://localhost:${PORT}`);
});
