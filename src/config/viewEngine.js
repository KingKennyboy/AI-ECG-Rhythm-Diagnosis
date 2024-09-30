const express = require('express');

function configViewEngine(app) {
    app.set('view engine', 'ejs');
    app.set('views', './views');
}

module.exports = configViewEngine;
