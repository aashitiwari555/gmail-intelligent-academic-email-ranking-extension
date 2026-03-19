console.log("Smart Gmail Sorter Loaded");

async function processEmails() {

    let emailRows = document.querySelectorAll("tr.zA");
    let emailData = [];

    for (let row of emailRows) {

    if (row.dataset.processed === "true") continue;

    let subjectElement = row.querySelector(".bog");

    if (!subjectElement) continue;

    if (subjectElement.querySelector(".ai-tag")) continue;

    let emailText = subjectElement.innerText;

    try {
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

        let intentTag = document.createElement("span");
        intentTag.innerText = "[" + data.intent + "] ";
        intentTag.style.color = "blue";
        intentTag.style.fontWeight = "bold";
        intentTag.classList.add("ai-tag");

        subjectElement.prepend(intentTag);

        if (data.priority === "High"){
            row.style.backgroundColor = "#ffdddd";
        }

        row.dataset.processed = "true";

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
