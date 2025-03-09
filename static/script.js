function fetchMovies() {
    let emotionNumber = document.getElementById("emotion").value;
    
    if (!emotionNumber) {
        alert("Please enter an emotion number.");
        return;
    }

    fetch('/get-movies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion_number: parseInt(emotionNumber) })
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        if (data.error) {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <h3 class="emotion-title">Detected Emotion: ${data.emotion}</h3>
                <p class="recommendation-title">Recommended Movies:</p>
                <ul class="movie-list">${data.movies.map(movie => `<li class="movie-item">${movie}</li>`).join('')}</ul>`;
        }
    })
    .catch(error => console.error('Error:', error));
}

async function uploadAndAnalyze() {
    const fileInput = document.getElementById('audioFile');
    const uploadStatus = document.getElementById('uploadStatus');
    const resultDiv = document.getElementById('result');

    if (!fileInput.files.length) {
        alert("Please select an audio file.");
        return;
    }

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);

    uploadStatus.innerHTML = 'Uploading and analyzing audio...';

    try {
        const response = await fetch('/analyze-audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <h3 class="emotion-title">Detected Emotion: ${data.emotion}</h3>
                <p class="recommendation-title">Recommended Movies:</p>
                <ul class="movie-list">${data.movies.map(movie => `<li class="movie-item">${movie}</li>`).join('')}</ul>`;
        }
        
        resultDiv.style.display = 'block'; // Show the result box
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = '<p class="error">Error processing audio file</p>';
        resultDiv.style.display = 'block'; // Show the result box
    } finally {
        uploadStatus.innerHTML = '';
    }
}

function createParticles() {
    const particleContainer = document.getElementById("particles");

    for (let i = 0; i < 50; i++) { // Increased the number of particles
        let particle = document.createElement("div");
        particle.classList.add("particle");
        
        let size = Math.random() * 10 + 5;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.bottom = `-${size}px`; // Start below the screen
        particle.style.animationDuration = `${Math.random() * 8 + 4}s`;
        particle.style.animationDelay = `${Math.random() * 2}s`; // Random delay for natural effect

        particleContainer.appendChild(particle);

        // Remove particles after animation
        setTimeout(() => particle.remove(), 12000);
    }
}

// Generate particles every 1.5 seconds for a continuous effect
setInterval(createParticles, 1500);

// Create some initial particles on load
window.onload = createParticles;
