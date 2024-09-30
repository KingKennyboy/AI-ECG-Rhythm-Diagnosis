function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  var addButton = document.getElementById("new-diagnosis-btn");  // Get the button element

  // Hide all tab contents and handle the button visibility
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Remove "active" class from all tab buttons
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the selected tab and add "active" class to the button that opened the tab
  document.getElementById(tabName).style.display = "flex";  // Use 'flex' to maintain container properties
  if (evt) {
    evt.currentTarget.className += " active";
  } else {
    // Manually set the active class on first load for DiagnosisResults
    if (tabName === 'DiagnosisResults' && tablinks.length > 0) {
      tablinks[0].className += " active";
    }
  }

  // Conditional visibility for the 'Add New Diagnosis' button
  if (tabName === "DiagnosisResults") {
    addButton.style.display = "inline-block";  // Show the button for DiagnosisResults tab
  } else {
    addButton.style.display = "none";  // Hide the button for other tabs
  }
}

document.addEventListener("DOMContentLoaded", function() {
  // Open the Diagnosis Results tab by default
  openTab(null, 'DiagnosisResults');  // Initialize tabs properly
});
