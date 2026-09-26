const fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"];

const DISPLAY = {
  "Iris-setosa": { common: "Setosa", latin: "Iris setosa" },
  "Iris-versicolor": { common: "Versicolor", latin: "Iris versicolor" },
  "Iris-virginica": { common: "Virginica", latin: "Iris virginica" },
};

fields.forEach((id) => {
  const slider = document.getElementById(id);
  const out = document.getElementById(`${id}_out`);
  slider.addEventListener("input", () => {
    out.textContent = parseFloat(slider.value).toFixed(1);
  });
});

const btn = document.getElementById("identify-btn");
const errorMsg = document.getElementById("error-msg");
const placeholder = document.getElementById("placeholder");
const specimenLabel = document.getElementById("specimen-label");
const commonName = document.getElementById("common-name");
const latinName = document.getElementById("latin-name");
const confidenceBlock = document.getElementById("confidence");

const artEls = {
  "Iris-setosa": document.getElementById("art-setosa"),
  "Iris-versicolor": document.getElementById("art-versicolor"),
  "Iris-virginica": document.getElementById("art-virginica"),
};

function showSpecies(species) {
  Object.entries(artEls).forEach(([key, el]) => {
    el.hidden = key !== species;
  });
}

btn.addEventListener("click", async () => {
  errorMsg.hidden = true;
  btn.disabled = true;
  btn.textContent = "Identifying…";

  const payload = {};
  fields.forEach((id) => {
    payload[id] = document.getElementById(id).value;
  });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    const top = data.prediction.species;
    placeholder.hidden = true;
    specimenLabel.hidden = false;
    commonName.textContent = DISPLAY[top]?.common ?? top;
    latinName.textContent = DISPLAY[top]?.latin ?? "";
    showSpecies(top);

    confidenceBlock.hidden = false;
    data.all.forEach((row) => {
      const key = row.species.replace("Iris-", "");
      const fill = document.getElementById(`bar-${key}`);
      const pct = document.getElementById(`pct-${key}`);
      if (fill && pct) {
        const percent = Math.round(row.confidence * 100);
        fill.style.width = `${percent}%`;
        pct.textContent = `${percent}%`;
      }
    });
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Identify specimen";
  }
});
