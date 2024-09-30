const request = require('supertest');
const session = require('express-session');
const express = require('express');
const sessionTest = require('supertest-session');

const app = express();
app.use(express.json());
app.use(session({
  secret: 'testsecret',
  resave: false,
  saveUninitialized: true,
  cookie: { secure: false }
}));

app.get('/dashboard', (req, res) => {
  if (req.session.user) {
    res.status(200).send("Dashboard Page");
  } else {
    res.redirect('/login');
  }
});

app.post('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) {
      res.status(500).send("Failed to logout");
    } else {
      res.clearCookie('connect.sid');
      res.redirect('/login');
    }
  });
});

describe('Dashboard Route Tests', () => {
  let testSession = null;

  beforeEach(function () {
    testSession = sessionTest(app);
  });

  test('POST /logout - should clear the user session and redirect', async () => {
    await testSession.post('/dashboard').send({ username: 'admin', password: 'admin' });
    const response = await testSession.post('/logout');
    expect(response.statusCode).toBe(302);
    expect(response.headers.location).toBe('/login');
  });
});
