document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const accuracyElem = document.getElementById('accuracy');
    const precisionElem = document.getElementById('precision');
    const recallElem = document.getElementById('recall');
    const f1ScoreElem = document.getElementById('f1_score');
    const barChartElem = document.getElementById('barChart').getContext('2d');
    const pieChartElem = document.getElementById('pieChart').getContext('2d');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(form);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            accuracyElem.textContent = data.accuracy.toFixed(2);
            precisionElem.textContent = data.precision.toFixed(2);
            recallElem.textContent = data.recall.toFixed(2);
            f1ScoreElem.textContent = data.f1_score.toFixed(2);

            const teamNames = data.top_50_teams.map(team => team.TeamName);
            const scores = data.top_50_teams.map(team => team.Score);

            // Bar Chart
            new Chart(barChartElem, {
                type: 'bar',
                data: {
                    labels: teamNames,
                    datasets: [{
                        label: 'Score',
                        data: scores,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Pie Chart
            new Chart(pieChartElem, {
                type: 'pie',
                data: {
                    labels: teamNames,
                    datasets: [{
                        label: 'Score',
                        data: scores,
                        backgroundColor: teamNames.map(() => `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.2)`),
                        borderColor: teamNames.map(() => `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 1)`),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed !== null) {
                                        label += context.parsed.toFixed(2);
                                    }
                                    return label;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error:', error));
    });
});
