# TheBloke/WizardCoder-Python-34B-V1.0-GPTQ Cog model

This is an implementation of the [TheBloke/WizardCoder-Python-34B-V1.0-GPTQQ](https://huggingface.co/TheBloke/WizardCoder-Python-34B-V1.0-GPTQ) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then run the git clone command at the end of the download-weights file

Then, you can run predictions:

    cog predict -i prompt="Tell me about AI"
