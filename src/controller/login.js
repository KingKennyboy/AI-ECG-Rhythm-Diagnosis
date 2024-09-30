const express = require('express');
const router = express.Router();

// Display the login form
router.get('/login', (req, res) => {
    res.render('login');
});

// Handle login form submission
router.post('/login', (req, res) => {
    const { userId, password } = req.body;
    if (userId === 'admin' && password === 'admin') {
        req.session.user = { id: userId, role: 'admin' };
        res.redirect('/dashboard');
    } else {
        res.render('login', { error: 'Invalid credentials!' });
    }
});

module.exports = router;