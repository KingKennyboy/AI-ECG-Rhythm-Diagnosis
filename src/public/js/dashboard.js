$(document).ready(function() {
    // When the New Diagnosis button is clicked, show the popup form
    $('#new-diagnosis-btn').click(function() {
        $('#new-diagnosis-popup').removeClass('hidden');
    });

    // Close the popup when the close button is clicked
    $('.close-btn').click(function() {
        $('#new-diagnosis-popup').addClass('hidden');
    });

    // Handle the form submission event
    $('#diagnosis-form').submit(function(event) {
        event.preventDefault();

        var formData = new FormData(this);
        $('#loading-bar').removeClass('hidden');
        $.ajax({
            type: 'POST',
            url: '/api/upload',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // Hide loading bar
                $('#loading-bar').addClass('hidden');

                if (response.outputFromPython.message && response.outputFromPython.message.trim() !== "") {
                    $('#message').text(response.outputFromPython.message).parent().removeClass('hidden');
                    $('#message').css("color","blue")
                    $('#patientName').text('N/A');
                    $('#rhythmResult').text('N/A');
                    $('#rhythmNote').text('N/A');
                    $('#rhythmDescription').text('N/A');
                    $('#new-diagnosis-popup').addClass('hidden');
                } else {
                    $('#DiagnosisResults').removeClass('hidden').css('display', 'block');
                    $('#new-diagnosis-popup').addClass('hidden');
                    $('#messageBlock').addClass('hidden');

                    var rhythmDetails = findRhythmDetails(response.outputFromPython.diagnosis_result);
                    if (rhythmDetails.matchFound) {
                        $('#patientName').text(response.outputFromPython.patient_name);
                        $('#rhythmResult').text(rhythmDetails.name);
                        $('#rhythmNote').text(rhythmDetails.note);
                        $('#rhythmDescription').text(rhythmDetails.description);

                        // $('#fileName').text(response.outputFromPython.filename); // Temporary presentation code
                        // $('#fileRhythm').text(response.outputFromPython.corresponding_rhythm); // Temporary presentation code

                        if (rhythmDetails.name.toLowerCase() === response.outputFromPython.corresponding_rhythm.toLowerCase()) {
                            $('.actualDiagnosis').css('background-color', '#a9ffa9');
                        } else {
                            $('.actualDiagnosis').css('background-color', '');
                        }
                    } else {
                        $('#rhythmNote').text('No matching rhythm found.');
                        $('#rhythmDescription').text('');
                    }
                }
            },
            error: function(xhr, status, error) {
                // Handle errors here
                console.error('Diagnosis submission failed: ' + error);
                $('#loading-bar').addClass('hidden');
                $('#message').text('Failed to process the file. Please try again.').parent().removeClass('hidden');
                $('#DiagnosisResults').addClass('hidden');
            }
        });
    });
    $('#loading-bar').click(function() {
        $(this).addClass('hidden');
    });
});

function findRhythmDetails(predictedRhythm) {
    const rhythm = definedRhythms.find(r => r.name.toLowerCase() === predictedRhythm.toLowerCase());
    return rhythm ? { matchFound: true, ...rhythm } : { matchFound: false, predicted_Rhythm: predictedRhythm };
}
