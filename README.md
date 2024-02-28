# SocialMediaSpamClassification
A work testing models for spam classification in social media comments using different libraries and models.


# How to run
## Existing Models
Start API "REST_API.py" and access index.html
 - API Keys required for youtube and facebook comments.
 - May need to install some python libraries (flask, bs4, deep_translator, googleapiclient, scrapy, )


## Train own models
Files are numbered.
1 - Merges all datasets in Datasets/ into one file (One dataset needs to be decompressed (Datasets/Fake News/Combinados.rar --> Contains a file with Fake.csv + True.csv))
2 - Prepares data and trains modes using a library (in filename - keras, sklearn, pytorch)
3 - Tests Models using test data
4 - Used for testing the models with comments from facebook, youtube or reddit (API keys required)
