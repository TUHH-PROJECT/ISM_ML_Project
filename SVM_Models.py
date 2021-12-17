from SVM_Fct import SVM_Classification

F1 = 'wavelet_avergae'
F2 = 'wavelet_energy'
F3 = 'wavelet_entropy'
F4 = 'wavelet_kurtosis'
F5 = 'wavelet_rms'
F6 = 'wavelet_skewness'
F7 = 'wavelet_std_deviation'
B = 'Basic'
S = 'Shearlet'
H = 'Hu'

#Best Combination for the Wavelet transform:
Wave = [F1, F2, F3, F4, F5, F6, F7]
acc = 0
Features = []
l1 = 0
l2 = 0
l3 = 0
l4 = 0
l5 = 0
l6 = 0
l7 = 0

for Feat in Wave:
    print('Loop1:', str(l1))
    l1 = l1+1
    acc_new = SVM_Classification(Feat)
    if acc_new > acc:
        acc = acc_new
        Features = [Feat]
    if l1 <= 6:
        for Feat2 in Wave[l1:]:
            print('Loop2:', str(l2))
            l2 = l2+1
            acc_new = SVM_Classification(Feat, Feature2=Feat2)
            if acc_new > acc:
                acc = acc_new
                Features = [Feat, Feat2]
            if l2+1 <= 6:
                for Feat3 in Wave[l2+1:]:
                    print('Loop3:', str(l3))
                    l3 = l3+1
                    acc_new = SVM_Classification(Feat, Feature2=Feat2, Feature3=Feat3)
                    if acc_new > acc:
                        acc = acc_new
                        Features = [Feat, Feat2, Feat3]
                    if l3+2 <= 6:
                        for Feat4 in Wave[l3+2:]:
                            print('Loop4:', str(l4))
                            l4 = l4+1
                            acc_new = SVM_Classification(Feat, Feature2=Feat2, Feature3=Feat3, Feature4=Feat4)
                            if acc_new > acc:
                                acc = acc_new
                                Features = [Feat, Feat2, Feat3, Feat4]
                            if l4+3 <= 6:
                                for Feat5 in Wave[l4+3:]:
                                    print('Loop5:', str(l5))
                                    l5 = l5+1
                                    acc_new = SVM_Classification(Feat, Feature2=Feat2, Feature3=Feat3, Feature4=Feat4, Feature5=Feat5)
                                    if acc_new > acc:
                                        acc = acc_new
                                        Features = [Feat, Feat2, Feat3, Feat4, Feat5]
                                    if l5+4 <= 6:
                                        for Feat6 in Wave[l5+4:]:
                                            print('Loop6:', str(l6))
                                            l6 = l6+1
                                            acc_new = SVM_Classification(Feat, Feature2=Feat2, Feature3=Feat3, Feature4=Feat4, Feature5=Feat5, Feature6=Feat6)
                                            if acc_new > acc:
                                                acc = acc_new
                                                Features = [Feat, Feat2, Feat3, Feat4, Feat5, Feat6]
                                            if l6+5 <= 6:
                                                for Feat7 in Wave[l6+5:]:
                                                    print('Loop7:', str(l7))
                                                    l7 = l7+1
                                                    acc_new = SVM_Classification(Feat, Feature2=Feat2, Feature3=Feat3, Feature4=Feat4, Feature5=Feat5, Feature6=Feat6, Feature7=Feat7)
                                                    if acc_new > acc:
                                                        acc = acc_new
                                                        Features = [Feat, Feat2, Feat3, Feat4, Feat5, Feat6, Feat7]       

print(acc)
print(Features)

#Result: Best accurancy is 0.61 and is achieved with all wavelet features

SVM_Classification(B, save = True, report_name = 'Report_Basic_SVM_RobustScaler.txt', model_name = 'Basic_SVM_RobustScaler.sav')
SVM_Classification(S, save = True, report_name = 'Report_Shearlet_SVM_RobustScaler.txt', model_name = 'Shearlet_SVM_RobustScaler.sav')
SVM_Classification(F1, save = True, report_name='Report_OptWavelet_RobustScaler.txt', model_name= 'OptWavelet_RobustScaler.sav', Feature2=F2,Feature3=F3,Feature4=F4,Feature5=F5,Feature6=F6, Feature7=F7)
SVM_Classification(H, save = True, report_name = 'Report_Hu_RobustScaler.txt', model_name = 'Hu_SVM_RobustScaler.sav')