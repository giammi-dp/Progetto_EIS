from pipeline import run
import matplotlib.pyplot as plt

case_name = 'BraTS-GLI-00000-000'
user_prompt = 'Generate a professional medical report'
image, report = run(case_name, user_prompt)
print(report)
plt.imshow(image)
plt.show()
