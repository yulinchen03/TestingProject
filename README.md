# Testing and Validation for AI-Intensive Systems (Group 7)

This repository contains an ONNX models for Rotterdam welfare system.
One of the models is a bad model and one is a good one. You need to guess which one.
Hints of how the models can be tested indicated in the current markdown file.
Good luck!

---

## Requirements
List the tools, libraries, and environments required to run the model. For example:
- Python 3.8 or higher
  Install it using:  
  ```bash
  pip install -r requirements.txt
  
## How to Run
Before running models, you need to specify the data you want to train your model for.
Then, for Model 1 and Model 2, just run Notebook <code>src/team_1/model1_training.ipynb</code>.
  
## Testing Hints for Model 1 and Model 2
Both models were tested using the same tests. The goal of one model was to
minimize the bias on these tests, while the goal of the other model was the opposite.

Both models were tested on the following criteria:
- **<u>General (based on the most (potentially) biased features)</u>**
  - **Gender Bias**: The model should not be biased based on the gender of the person.
  - **Age Bias**: The model should not be biased based on the age of the person.
  - **History of Development**: The model should not be biased based on the history of the person, based on his development plan.
- **<u>Specific (based on the specific cases from Wired articles)</u>**
  - **Single Mother**: we tried to decrease the bias for single mothers.
  - **Immigrant Worker**: we compared immigrants with native workers and ensured the decrease of bias.
  - **High Risk Individual**: we tried to decrease the bias for high-risk individuals, enduring equal choice of being checked. By high risk individual we mean individuals from Wired articles who considered high risk by the original algorithm.
