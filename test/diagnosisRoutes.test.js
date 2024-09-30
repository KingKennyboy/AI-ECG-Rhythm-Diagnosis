// Keeping initial requires that don't need to be isolated at the top
const request = require('supertest');

jest.mock('multer', () => {
  const multer = () => ({
    single: jest.fn(() => (req, res, next) => {
      req.file = req.body.shouldHaveFile ? {
        buffer: Buffer.from('some data', 'utf-8'),
        originalname: 'test.csv'
      } : null;
      next();
    })
  });
  multer.memoryStorage = jest.fn(); // Ensure memoryStorage is mocked as a function of multer
  return multer; // Return the function expected by your route setup
});

jest.mock('child_process', () => ({
  spawn: jest.fn()
}));

jest.mock('../src/config/db', () => ({
  query: jest.fn((sql, params, callback) => callback(null, { insertId: 1 }))
}));

describe('POST /upload', () => {
    let app;

    beforeEach(() => {
        jest.resetAllMocks();
        // Setup express app within each test scope to avoid state leakage
        const express = require('express');
        app = express();
        app.use(express.json());
        app.use(express.urlencoded({ extended: true }));

        const routes = require('../src/routes/diagnosisRoutes');
        app.use('/', routes);
    });

    test('responds with 400 if no file is uploaded', async () => {
        const response = await request(app)
            .post('/upload')
            .send({ patientName: 'John Doe', modelType: 'modelXYZ', shouldHaveFile: false });

        expect(response.status).toBe(400);
        expect(response.text).toContain('No CSV file uploaded.');
    });

});