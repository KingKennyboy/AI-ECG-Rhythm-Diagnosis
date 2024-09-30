const express = require('express');
const router = express.Router();


// Serve the login page directly on the root URL
router.get('/', (req, res) => {
    res.render('login');  // Directly render the login view
});

// Additional routes, if needed
router.get('/about', (req, res) => {
    res.send('About Us page content here');
});

module.exports = router;
