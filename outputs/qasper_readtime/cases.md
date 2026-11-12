# Case Analysis

Scope: sample100, grouping section. Rankings are top-5. The unit lists below are the top-5 comprehension units ranked by unit-question embedding similarity; GOLD marks that the unit's source paragraph is gold evidence for that question (it does not guarantee the fused passage ranking kept it).

## Positive: M5 improves All-Evidence@5 over M2

### 50be4a737dc0951b35d139f51075011095d77f2a — What background knowledge do they leverage?

- gold passages: 1503.00841::s01::p000, 1503.00841::s06::p000, 1503.00841::s06::p002
- All@5: M2=0.000 → M3=0.000 → M4=0.500 → M5=0.500
- units in top-5:
  - `1503.00841::u000` [GOLD] The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust. (src: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s00::p005, 1503.00841::s01::p000)
  - `1503.00841::u010` [not-gold] Unlike the maximum entropy regularization term, KL divergence handles unbalanced corpora better by using the reference class distribution instead of assuming uniformly distributed categories. (src: 1503.00841::s13::p003, 1503.00841::s13::p004)
  - `1503.00841::u009` [not-gold] The maximum entropy regularization term controls unbalance without extra knowledge, but its uniform-category assumption creates degraded performance when violated. (src: 1503.00841::s13::p002, 1503.00841::s13::p003)
  - `1503.00841::u002` [not-gold] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (src: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)
  - `1503.00841::u007` [not-gold] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (src: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)
### 02428a8fec9788f6dc3a86b5d5f3aa679935678d — How do they define robustness of a model?

- gold passages: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s04::p000, 1503.00841::s07::p001, 1503.00841::s09::p000, 1503.00841::s09::p002
- All@5: M2=0.000 → M3=0.000 → M4=0.500 → M5=0.500
- units in top-5:
  - `1503.00841::u000` [GOLD] The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust. (src: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s00::p005, 1503.00841::s01::p000)
  - `1503.00841::u006` [not-gold] LDA selects predictive labeled features without considering class balance, and GE-FL does not control this issue, so auxiliary regularization terms promote the model significantly. (src: 1503.00841::s11::p001, 1503.00841::s11::p002)
  - `1503.00841::u009` [not-gold] The maximum entropy regularization term controls unbalance without extra knowledge, but its uniform-category assumption creates degraded performance when violated. (src: 1503.00841::s13::p002, 1503.00841::s13::p003)
  - `1503.00841::u007` [not-gold] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (src: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)
  - `1503.00841::u002` [GOLD] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (src: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)
### 4986f420884f917d1f60d3cea04dc8e64d3b5bf1 — Which additional latent variables are used in the model?

- gold passages: 1603.01514::s00::p006, 1603.01514::s03::p000
- All@5: M2=0.500 → M3=0.500 → M4=1.000 → M5=1.000
- units in top-5:
  - `1603.01514::u000` [GOLD] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u009` [GOLD] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)
### 6568a31241167f618ef5ede939053feaa2fb0d7e — Do they add one latent variable for each language pair in their Bayesian model?

- gold passages: 1603.01514::s03::p000
- All@5: M2=0.000 → M3=0.000 → M4=1.000 → M5=1.000
- units in top-5:
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u009` [GOLD] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)
### b85fc420eb2f77f6f14f375cc1fcc5155eb5c0a8 — Does the paper clearly establish that the challenges listed here exist in this dataset and task?

- gold passages: 1605.03481::s00::p002
- All@5: M2=0.000 → M3=0.000 → M4=1.000 → M5=1.000
- units in top-5:
  - `1605.03481::u006` [GOLD] The noisy Twitter phenomena of slang, misspellings, emojis and special characters motivate the character based approach, whose predictions show robustness to spelling mistakes and effective interpretation of emojis and o (src: 1605.03481::s00::p002, 1605.03481::s06::p004)
  - `1605.03481::u005` [not-gold] The word-level baseline's whitespace tokenization and `UNK' grouping instantiate the preprocessing and unseen-word limitations attributed to word-based methods. (src: 1605.03481::s01::p000, 1605.03481::s03::p000)
  - `1605.03481::u001` [not-gold] Because the word-level baseline stores only the 20K most common words and groups the rest under `UNK', tweet2vec's significantly better RW test set performance supports its advantage on out-of-vocabulary words. (src: 1605.03481::s03::p000, 1605.03481::s06::p001)
  - `1605.03481::u003` [not-gold] The final dataset's 2039 distinct hashtags define the softmax output layer size L used to compute posterior hashtag probabilities from the tweet embedding. (src: 1605.03481::s02::p005, 1605.03481::s04::p000)
  - `1605.03481::u007` [not-gold] The assumption that posts with the same hashtags should have close embeddings is operationalized by optimizing categorical cross-entropy loss between predicted and true hashtags. (src: 1605.03481::s00::p003, 1605.03481::s02::p007)
### b91671715ad4fad56c67c28ce6f29e180fe08595 — What other tasks do they test their method on?

- gold passages: 1605.03481::s06::p000
- All@5: M2=0.000 → M3=0.000 → M4=1.000 → M5=1.000
- units in top-5:
  - `1605.03481::u006` [not-gold] The noisy Twitter phenomena of slang, misspellings, emojis and special characters motivate the character based approach, whose predictions show robustness to spelling mistakes and effective interpretation of emojis and o (src: 1605.03481::s00::p002, 1605.03481::s06::p004)
  - `1605.03481::u005` [not-gold] The word-level baseline's whitespace tokenization and `UNK' grouping instantiate the preprocessing and unseen-word limitations attributed to word-based methods. (src: 1605.03481::s01::p000, 1605.03481::s03::p000)
  - `1605.03481::u001` [not-gold] Because the word-level baseline stores only the 20K most common words and groups the rest under `UNK', tweet2vec's significantly better RW test set performance supports its advantage on out-of-vocabulary words. (src: 1605.03481::s03::p000, 1605.03481::s06::p001)
  - `1605.03481::u000` [GOLD] tweet2vec uses hashtag-prediction supervision to train character-based tweet embeddings and outperforms the word model on held-out hashtag prediction. (src: 1605.03481::s00::p003, 1605.03481::s00::p004, 1605.03481::s06::p000, 1605.03481::s06::p001)
  - `1605.03481::u007` [not-gold] The assumption that posts with the same hashtags should have close embeddings is operationalized by optimizing categorical cross-entropy loss between predicted and true hashtags. (src: 1605.03481::s00::p003, 1605.03481::s02::p007)

## Negative: units or triples displace correct evidence

### 6bf5620f295b5243230bc97b340fae6e92304595 — What baseline model is used?

- gold passages: 1603.01514::s06::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)
### 747b847d687f703cc20a87877c5b138f26ff137d — Which parallel corpora are used?

- gold passages: 1603.01514::s08::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)
### 50cc6c5f2dcf5fb87b56007f6a825fa7c90b64ed — What does an individual model consist of?

- gold passages: 1603.01514::s02::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)
### 9cf070d6671ee4a6353f79a165aa648309e01295 — What is the size of the parallel corpus used to train the model constraints?

- gold passages: 1603.08594::s05::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.08594::u000` [not-gold] The Dual Decomposition (DD) based algorithm improved PP-attachment accuracy by 10% over the MSTParser baseline on 100 parallel sentences from the Hindi-English Tourism parallel corpus. (src: 1603.08594::s00::p012, 1603.08594::s05::p001, 1603.08594::s05::p002)
  - `1603.08594::u003` [not-gold] The DD algorithm's input is a parallel English-Hindi sentence pair with word alignments, while GIZA++ word-alignment errors may cascade and cause the system to underperform. (src: 1603.08594::s02::p000, 1603.08594::s05::p004)
  - `1603.08594::u006` [not-gold] The DD algorithm was evaluated on 100 parallel sentences with manually annotated PP-attachment ambiguity, where it marked 64 attachments correctly in the best case. (src: 1603.08594::s05::p001, 1603.08594::s05::p002)
  - `1603.08594::u004` [not-gold] The Project Algorithm predicts target-language trees using source-language parse trees and word alignments, and its projected paths are supplied by two classifiers in a pipeline. (src: 1603.08594::s03::p003, 1603.08594::s04::p000)
### 87bc6f83f7f90df3c6c37659139b92657c3f7a38 — How does enforcing agreement between parse trees work across different languages?

- gold passages: 1603.08594::s02::p001
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.08594::u004` [not-gold] The Project Algorithm predicts target-language trees using source-language parse trees and word alignments, and its projected paths are supplied by two classifiers in a pipeline. (src: 1603.08594::s03::p003, 1603.08594::s04::p000)
  - `1603.08594::u000` [not-gold] The Dual Decomposition (DD) based algorithm improved PP-attachment accuracy by 10% over the MSTParser baseline on 100 parallel sentences from the Hindi-English Tourism parallel corpus. (src: 1603.08594::s00::p012, 1603.08594::s05::p001, 1603.08594::s05::p002)
  - `1603.08594::u003` [not-gold] The DD algorithm's input is a parallel English-Hindi sentence pair with word alignments, while GIZA++ word-alignment errors may cascade and cause the system to underperform. (src: 1603.08594::s02::p000, 1603.08594::s05::p004)
  - `1603.08594::u006` [not-gold] The DD algorithm was evaluated on 100 parallel sentences with manually annotated PP-attachment ambiguity, where it marked 64 attachments correctly in the best case. (src: 1603.08594::s05::p001, 1603.08594::s05::p002)
### 26327ccebc620a73ba37a95aabe968864e3392b2 — what aspects of conversation flow do they look at?

- gold passages: 1604.03114::s00::p001, 1604.03114::s02::p003, 1604.03114::s03::p001
- All@5: M2=0.500 → M3=0.500 → M4=0.500 → M5=0.500; Any@5: M2=1.000 → M5=0.500
  - `1604.03114::u009` [not-gold] The paper's focus on conversational flow extends previous influence studies in relatively unstructured political settings by using highly structured Oxford-style debates that control for confounding factors. (src: 1604.03114::s00::p002, 1604.03114::s04::p002, 1604.03114::s05::p002)
  - `1604.03114::u000` [GOLD] The framework's conversational flow patterns are evaluated on 108 IQ2 debates where pre- and post-debate votes define winners, yielding Flow: 63% and Flow*: 65% accuracy. (src: 1604.03114::s00::p001, 1604.03114::s01::p001, 1604.03114::s01::p002, 1604.03114::s03::p005)
  - `1604.03114::u007` [not-gold] Flow features outperform length and BOW baselines despite excluding the concluding round, where winning teams tend to receive more applause. (src: 1604.03114::s01::p003, 1604.03114::s03::p005)
  - `1604.03114::u008` [not-gold] The basic unigram-based definition of discussion points limits the robustness of idea flow analysis because it does not account for context or semantic sense. (src: 1604.03114::s02::p006, 1604.03114::s05::p001)
  - `1604.03114::u006` [not-gold] Audience features use applause and laughter counts as rough outcome indicators because winning teams tend to receive more laughter during discussion and more applause during conclusion. (src: 1604.03114::s01::p003, 1604.03114::s03::p003)

## Units retrieved but not touching gold

### 76121e359dfe3f16c2a352bd35f28005f2a40da3 — What NLP tasks do they consider?

- gold: 1503.00841::s05::p000, 1503.00841::s06::p000
  - `1503.00841::u007` [not-gold] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (src: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)
  - `1503.00841::u010` [not-gold] Unlike the maximum entropy regularization term, KL divergence handles unbalanced corpora better by using the reference class distribution instead of assuming uniformly distributed categories. (src: 1503.00841::s13::p003, 1503.00841::s13::p004)
  - `1503.00841::u006` [not-gold] LDA selects predictive labeled features without considering class balance, and GE-FL does not control this issue, so auxiliary regularization terms promote the model significantly. (src: 1503.00841::s11::p001, 1503.00841::s11::p002)
  - `1503.00841::u002` [not-gold] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (src: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)
### 6bf5620f295b5243230bc97b340fae6e92304595 — What baseline model is used?

- gold: 1603.01514::s06::p000
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
### 747b847d687f703cc20a87877c5b138f26ff137d — Which parallel corpora are used?

- gold: 1603.01514::s08::p000
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
### 50cc6c5f2dcf5fb87b56007f6a825fa7c90b64ed — What does an individual model consist of?

- gold: 1603.01514::s02::p000
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)

## Overreach rejected by the verifier

- GE-FL is the baseline that leverages labeled features as prior knowledge, and the paper addresses the robustness problem on top of GE-FL.  
  reason: The concise_text is a restatement of s01:p000, and the triple's predicate 'requires' is not supported: GE-FL is the baseline/method on top of which the robustness problem is addressed, not something requiring that problem.
- The multilingual model uses word alignments between sentences in a parallel corpus to couple monolingual models, but only 8% of English arguments and 17% of German arguments are aligned.  
  reason: Passages state the model uses alignments and that only 8%/17% of arguments are aligned, but do not support that these percentages 'limit' the model's use of alignments as a causal relation; this is inferred.
- Because crosslingual parameters are ignored when trained parameters parse monolingual data, information coming from the CLVs acts as a regularizer for the monolingual models.  
  reason: This is essentially a restatement of p009: crosslingual parameters are ignored at parsing and thus CLV information acts as a regularizer. Reject as single-paragraph paraphrase.