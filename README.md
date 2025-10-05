# Thesis
Files created for my thesis, "ANÁLISE E DESENVOLVIMENTO DE MODELOS PARA AGENDAMENTO DE EXAMES EM MEDICINA NUCLEAR: UM ESTUDO DE CASO". On the topic of operational research on Flexible Multi-Resource Job-Shop with No-Wait

Any of the models in these folder are ready to use. To modify them you'll need to go into Exam_Holder_Final and at the bottom build a rec array with the exams you want to perform in a given day, with pro_i being the number of exams of each type to perform.
In each one of the Models you'll need to modify the pro_i array that is passed to the main function. For the second problem you'll need to also define the cutoff date at which exams are counted towards the objective function.
Everything elso should be done "automatically", other variables you may want to change are Lk, freeze_crit, temp_red, punish, ratio, and rec_max_p but you'll need to directly change those in the files.
