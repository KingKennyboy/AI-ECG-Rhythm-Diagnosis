const request = require('supertest');
const express = require('express');
const session = require('express-session');
const loginRouter = require('../src/controller/login'); // Update with the actual path to your login router

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(session({
  secret: 'test_secret',
  resave: false,
  saveUninitialized: true,
  cookie: { secure: false }
}));

app.use(loginRouter);

describe('Login Routes', () => {
    let agent;

    beforeEach(() => {
        agent = request.agent(app); // Create a new agent to maintain session state
    });

    test('POST /login with correct credentials should redirect to dashboard', async () => {
        const response = await agent.post('/login').send({
            userId: 'admin',
            password: 'admin'
        });
        expect(response.status).toBe(302);
        expect(response.headers.location).toBe('/dashboard');
    });
});

