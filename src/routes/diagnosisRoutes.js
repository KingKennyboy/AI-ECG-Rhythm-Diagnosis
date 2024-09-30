const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });
const connection = require('../config/db');

router.post('/upload', upload.single('csvFile'), (req, res) => {
    const { patientName, modelType } = req.body;
    const csvFile = req.file;

    if (!csvFile) {
        return res.status(400).send('No CSV file uploaded.');
    }

    const pythonProcess = spawn('python', [
        'trainingModel/rhythmDiagnosis.py',
        patientName,
        modelType,
        // csvFile.originalname // temporary code
    ]);

    let fullOutput = '';
    pythonProcess.stdin.write(csvFile.buffer);
    pythonProcess.stdin.end();

    pythonProcess.stdout.on('data', (data) => {
        fullOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ message: "Python script failed to run" });
        }

        const jsonStart = fullOutput.indexOf('---JSON_START---') + '---JSON_START---'.length;
        const jsonEnd = fullOutput.indexOf('---JSON_END---');
        const jsonString = fullOutput.substring(jsonStart, jsonEnd).trim();

        try {
            const outputFromPython = JSON.parse(jsonString);
            console.log("Output From Python", outputFromPython);
            console.log("message:", outputFromPython.message)
            return res.status(200).json({
                outputFromPython: outputFromPython
            });

            // const query = 'INSERT INTO diagnoses (patient_name, diagnosis_result, model, diagnosis_date) VALUES (?, ?, ?, ?)';
            // connection.query(query, [outputFromPython.patient_name, outputFromPython.diagnosis_result, outputFromPython.model, diagnosis_date], (err, results) => {
            //     if (err) {
            //         console.error('Failed to insert data: ' + err);
            //         return res.status(500).json({ message: "Failed to insert data into database" });
            //     }
            //     console.log('Data inserted, ID:', results.insertId);
            //     res.status(200).json({
            //         outputFromPython: outputFromPython,
            //         diagnosisDate: diagnosis_date
            //     });
            // });
        } catch (error) {
            console.error("Error parsing JSON from Python script:", error);
            res.status(500).json({ message: "Error processing files" });
        }
    });

    pythonProcess.on('error', (error) => {
        console.error('Failed to spawn Python process', error);
        res.status(500).send('Internal Server Error');
    });
});

module.exports = router;
