console.log("Smart Gmail Sorter Loaded");

async function processEmails() {

    let emailRows = document.querySelectorAll("tr.zA");
    let emailData = [];

    for (let row of emailRows) {

        let subjectElement = row.querySelector(".bog");

        if (!subjectElement) continue;

        let emailText = subjectElement.innerText;

        try {
            // Local FastAPI backend used for ML predictions
            let response = await fetch("http://127.0.0.1:8000/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    email_text: emailText
                })

            });

            let data = await response.json();

            console.log("Prediction:", data);

            // add intent label
            let intentTag = document.createElement("span");
            intentTag.innerText = "[" + data.intent + "] ";
            intentTag.style.color = "blue";
            intentTag.style.fontWeight = "bold";

            subjectElement.prepend(intentTag);

            // highlight urgent emails
            if(data.urgency_score > 1.5){
                row.style.backgroundColor = "#ffdddd";
            }

            emailData.push({
                row: row,
                score: data.urgency_score
            });

        } catch(err) {
            console.error(err);
        }
    }

    // sort by urgency score
    emailData.sort((a,b) => b.score - a.score);

    let container = emailRows[0].parentNode;

    emailData.forEach(item => {
        container.appendChild(item.row);
    });
}

setTimeout(processEmails, 4000);
