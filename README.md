# Abstract

## Static Analysis Tools (SATs) have already become a necessity in modern software systems, while the excess of unactionable warnings reported by SATs severely hinders the usability of SATs. To address this problem, Machine Learning (ML) techniques are commonly used for Actionable Warning Identification (AWI), where one key problem is how to represent warnings well for AWI. The traditional ML-based AWI approaches mainly use hand-engineered features to capture the statistical information of warnings for AWI, which extremely relies on domain knowledge. Recently, the state-of-the-art ML-based AWI approaches have demonstrated that learning the lexical information from the warning-related source code can better represent warnings for AWI. However, such warning representation could miss the syntactic information of warnings. In this paper, we propose an AWI approach via source code representation, which aims to capture the lexical and syntactic information for AWI from the warning-related source code. Specifically, our approach first performs the program slicing to obtain the warning-related source code. Subsequently, our approach designs an adjustment algorithm to make the warning-related source code satisfy the syntax compilation. Next, our approach constructs the statement-level Abstract Syntax Tree (AST) for the warning-related source code. Finally, our approach introduces a novel source code representation technique to automatically learn the lexical and syntactic information from this statement-level AST as a feature vector. Based on feature vectors of labeled warnings, our approach trains a ML model for AWI. The experimental evaluation on 56 releases from five large-scale and open-source projects shows that compared to four state-of-the-art ML-based AWI approaches, our approach can achieve the top-ranked AUC in both within-project and cross-project AWI.

# Dataset

## When decompressing "Dataset.zip", you can see five folders, where each fold is corresponding to a project. In each project, there are several releases. Each release has five files ".pkl", which is corresponding to the warning representation in all warnings of the current release.
<img width="168" alt="image" src="https://user-images.githubusercontent.com/18481003/204467221-7b6a69d5-2ebf-443a-bc03-72a79c84e274.png">
+ ast.pkl: the statement-level AST of the warning. Such information is used for our approach.
+ byteToken.pkl: the program slicing results of warnings, which consist of bytecode. Such information is used for SlicingLSTM.
+ fiveLine.pkl: the upper and lower five lines of the warning lines. Such information is used for CodeCNN.
+ func.pkl: the method where the warning.
+ sliceToken.pkl: the program slicing results of warnings, which consist of tokens. Such information is used for SlicingBow.
+ metric.xls: the warning representation of HFModel in all warnings of the current project. Such information is used for HFModel.

# Script

+ astEncode: it is used to construct the statement-level AST.
+ model: the ML and DL models used for our approach.
+ prepareData: the data preprocessing.
+ trainModel: the detailed implement details of models.
