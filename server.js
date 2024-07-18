const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const port = 5000;

// Enable CORS
app.use(cors());

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Function to analyze sentiment using Python script
function analyzeSentiment(text, callback) {
  const process = spawn('python', ['analyze_sentiment.py']);
  let result = '';

  process.stdout.on('data', (data) => {
    result += data.toString();
  });

  process.stderr.on('data', (data) => {
    console.error(`Error: ${data}`);
  });

  process.on('close', (code) => {
    if (code !== 0) {
      callback(new Error('Python script exited with code ' + code));
    } else {
      callback(null, JSON.parse(result));
    }
  });

  process.stdin.write(text);
  process.stdin.end();
}

// API endpoint for sentiment analysis
app.post('/analyze', (req, res) => {
  const text = req.body.text;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }

  analyzeSentiment(text, (error, sentiment) => {
    if (error) {
      console.error('Error analyzing sentiment:', error);
      return res.status(500).json({ error: 'Internal server error' });
    }
    var pos = sentiment['pos']
    var neg = sentiment['neg']
    var comp = sentiment['compound']
    if (comp > 0){
      pos = pos + comp
    }
    if (comp < 0){
      neg = neg + (-1 * comp)
    }
    sentiment['pos'] = pos
    sentiment['neg'] = neg
    res.json({ sentiment });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Sentiment analysis server running at http://localhost:${port}`);
});
