load fisheriris
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
N = size(meas,1);

lda = fitcdiscr(meas(:,1:2),species);
ldaClass = resubPredict(lda);

ldaResubErr = resubLoss(lda)

[ldaResubCM,grpOrder] = confusionmat(species,ldaClass);
ldaClass
species