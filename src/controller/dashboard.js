const express = require('express');
const router = express.Router();
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');

// Configure storage for the uploaded files
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, '..', '..', 'dataset', 'userECGUnprocessed')) // Adjusting path to correct location
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname)
  }
});

// Filter to check if uploaded file is a CSV
const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'text/csv') {
    cb(null, true);
  } else {
    cb(new Error('Only CSV files are allowed!'), false);
  }
};

const upload = multer({ storage: storage, fileFilter: fileFilter });

// GET dashboard page
router.get('/dashboard', (req, res) => {
  console.log(req.session.user); // Check session info
  if (req.session.user) {
    res.render('dashboard');
  } else {
    res.redirect('/login');
  }
});

// POST to handle new diagnosis
router.post('/new-diagnosis', upload.single('diagnosisFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }
  console.log('Received file:', req.file.filename);

  // Call Python script to process the uploaded CSV
  exec(`python diagnose.py ${path.join(__dirname, 'uploads', req.file.filename)}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return res.status(500).send(stderr);
    }
    console.log('Python Script Output:', stdout);
    res.send(stdout);
  });
});

router.post('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) {
      return res.redirect('/dashboard');
    }
    console.log("logged out")
    res.clearCookie('connect.sid'); // Assuming session cookie name is 'connect.sid'
    res.redirect('/login');
  });
});

module.exports = router;