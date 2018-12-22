# Clinical Notes Model
medaCy comes equipped with a powerful model for entity extraction from clinical notes

# Generalization Ability (TODO: needs formatting)
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
                Drug  0.8576  0.7907  0.8228    0.9525  0.8680  0.9083
            Strength  0.8732  0.8418  0.8572    0.9772  0.9340  0.9552
            Duration  0.6555  0.3624  0.4668    0.9187  0.5079  0.6542
               Route  0.9259  0.8432  0.8826    0.9701  0.8782  0.9219
                Form  0.8473  0.7499  0.7957    0.9613  0.8435  0.8986
                 Ade  0.1117  0.0352  0.0535    0.3990  0.1232  0.1883
              Dosage  0.7478  0.7057  0.7262    0.9152  0.8571  0.8852
              Reason  0.3428  0.2837  0.3105    0.5306  0.4263  0.4728
           Frequency  0.6767  0.6401  0.6579    0.9672  0.8886  0.9262
                      ------------------------------------------------
     Overall (micro)  0.7905  0.7137  0.7502    0.9243  0.8235  0.8710
     Overall (macro)  0.7786  0.6981  0.7343    0.9152  0.8099  0.8569
| Entities | Strict | Lenient |
| :-------: | :----------------: |:-------------:|
|Drug| | |  |
|Strength|  |  |
|Duration | |  |
|Route | |  |
|Form | |  |
|ADE | |  |
|Dosage | |  |
|Reason | |  |
|Frequency | |  |
