const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const session = require('express-session');

// Importing routers
const webRouter = require('./routes/web');
const loginRouter = require('./controller/login');
const dashboardRoutes = require('./controller/dashboard');
const diagnosisRoutes = require('./routes/diagnosisRoutes');  // Make sure the path matches where you save this file

const app = express();

// Set up EJS for templating
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware to parse the body of HTTP requests
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());  // Enable JSON body for API calls

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// // Cors middleware for cross-origin requests
// app.use(cors());

// Session middleware
app.use(session({
    secret: 'your-secret-key',  // Change to a random, secret string in production
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: false,  // Set to true if using https
        maxAge: 1000 * 60 * 60 * 24  // 24 hours
    }
}));

// Registering routers
app.use('/', webRouter);
app.use('/', loginRouter);
app.use('/', dashboardRoutes);
app.use('/api', diagnosisRoutes);  // Mounting the diagnosis routes under /api

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
