
    print("generating scores with 3ft model...")
    for row in dataset.itertuples():
        sentence = row[1]
        # print(sentence)
        score = get_empathy(sentence, empathy_model_3)
        print("score (3ft): ", str(score))
        score_numeric = label2int[score]
        # print("score numeric: ", str(score_numeric))
        scores_3.append(score_numeric)
        
    # add the scores to the CSV file
    data = pd.read_csv(score_data_path)
    data['NEWempathy_score_triple'] = pd.Series(scores_3)
    data.to_csv(score_data_path, index=False)
    
    # add the scores to the CSV file
    # data = pd.read_csv(score_data_path)
    # data['empathy_score_1FT_sentence'] = pd.Series(scores_1)
    
    print("generating scores with 2ft model...")
    scores_2 = []
    for row in dataset.itertuples():
        sentence = row[1]
        score = get_empathy(sentence, empathy_model_3)
        print("score: ", str(score))
        score_numeric = label2int[score]
        # print("score numeric: ", str(score_numeric))
        scores_2.append(score_numeric)
        
    # # # add the scores to the CSV file
    # data = pd.read_csv(score_data_path)
    # data['empathy_score'] = scores
    # data.to_csv(score_data_path, index=False)
    
    # add the scores to the CSV file
    data = pd.read_csv(score_data_path)
    data['NEWempathy_score_2'] = pd.Series(scores_2)

    data.to_csv(score_data_path, index=False)