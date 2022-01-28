library("NLP")
library("twitteR")
library("syuzhet")
library("tm")
library("SnowballC")
library("stringi")
library("topicmodels")
library("syuzhet")
library("ROAuth")

# Authonitical keys
consumer_key <- "XXX"
consumer_secret <- "XXX"
access_token <- "XXX"
access_secret <- "XXX"

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

Ozempic <- searchTwitter("Ozempic", n=10000,lang = "en")
Rybelsus <- searchTwitter("Rybelsus", n=1000,lang = "en")

Ozempic_tweets <- twListToDF(Ozempic)
Rybelsus_tweets <- twListToDF(nn1)

#convert all text to lower case
femininity_text<- tolower(femininity_text)
masculine_text<- tolower(masculine_text)
feminine_text<- tolower(feminine_text)

# Replace blank space ("rt")
Ozempic_text <- gsub("rt", "", Ozempic_text)
Rybelsus_text <- gsub("rt", "", Rybelsus_text)

# Replace @UserName
Rybelsus_text <- gsub("@\\w+", "", Rybelsus_text)


# Remove punctuation
Ozempic_text <- gsub("[[:punct:]]", "", Ozempic_text)


# Remove links
Ozempic_text <- gsub("http\\w+", "", Ozempic_text)
Rybelsus_text <- gsub("http\\w+", "", Rybelsus_text)

# Remove tabs
Ozempic_text <- gsub("[ |\t]{2,}", "", Ozempic_text)
Rybelsus_text <- gsub("[ |\t]{2,}", "", Rybelsus_text)


# Remove blank spaces at the beginning
Ozempic_text <- gsub("^ ", "", Ozempic_text)
Rybelsus_text <- gsub("^ ", "", Rybelsus_text)
masculine_text <- gsub("^ ", "", masculine_text)
feminine_text <- gsub("^ ", "", feminine_text)

femt <- gsub("ð+", "", femt)
# Remove blank spaces at the end
Ozempic_text <- gsub(" $", "", Ozempic_text)
Rybelsus_text <- gsub(" $", "", Rybelsus_text)

#getting emotions using in-built function
mysentiment_Ozempic<-get_nrc_sentiment((Ozempic_text))
mysentiment_Rybelsus<-get_nrc_sentiment((Rybelsus_text))

#calculationg total score for each sentiment
Sentimentscores_Ozempic<-data.frame(colSums(mysentiment_Ozempic[,]))
Sentimentscores_Rybelsus<-data.frame(colSums(mysentiment_Rybelsus[,]))

names(Sentimentscores_Ozempic)<-"Score"
Sentimentscores_Ozempic<-cbind("sentiment"=rownames(Sentimentscores_Ozempic),Sentimentscores_Ozempic)
rownames(Sentimentscores_Ozempic)<-NULL

names(Sentimentscores_Rybelsus)<-"Score"
Sentimentscores_Rybelsus<-cbind("sentiment"=rownames(Sentimentscores_Rybelsus),Sentimentscores_Rybelsus)
rownames(Sentimentscores_Rybelsus)<-NULL



#plotting the sentiments with scores
ggplot(data=Sentimentscores_Ozempic,aes(x=sentiment,y=Score))+geom_bar(aes(fill=sentiment),stat = "identity")+
  theme(legend.position="none")+
  xlab("Sentiments")+ylab("scores")+ggtitle("Sentiments of people behind the tweets on Ozempic")

library(ggplot2)


ggplot(data=Sentimentscores_Rybelsus,aes(x=sentiment,y=Score))+geom_bar(aes(fill=sentiment),stat = "identity")+
  theme(legend.position="none")+
  xlab("Sentiments")+ylab("scores")+ggtitle("Sentiments of people behind the tweets on Rybelsus")


