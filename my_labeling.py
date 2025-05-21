__authors__ = [1630717,1631990, 1632068, 1638180]
__group__ = 12

import numpy as np

from utils_data import *
from Kmeans import *
import time as t
from KNN import *
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
    train_imgs_grey, train_class_labels, train_color_labels, test_imgs_grey, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def Retrieval_by_color(imgs, labels_c, colors, dic_percents):
        idxs = []
        for i, label in enumerate(labels_c):
            app = True
            for color in colors:
                if (color not in label):
                    app = False
            if app == True:
               idxs.append(i)
        idxs = np.array(idxs)
        ordena = np.empty((idxs.shape[0],))
        for m,n in enumerate(idxs):
            media = 0
            for c in colors:
                media += dic_percents[n][c]
            ordena[m] = media
        idxs = idxs[np.argsort(ordena)][::-1]
        print(dic_percents[idxs])
        return imgs[idxs]


    def Retrieval_by_shape(imgs, labels_s, shapes, neighbors):
        idxs = np.array([], dtype=int)
        for shape in shapes:
            idxs = np.append(idxs, np.where(labels_s == shape))
        ordena = np.empty((idxs.shape[0],))
        for m,n in enumerate(idxs):
            ordena[m] = np.where(neighbors[n]==labels_s[n])[0].shape[0]
        idxs = idxs[np.argsort(ordena)][::-1]
        return imgs[idxs]


    def Retrieval_combined(imgs, labels_c, labels_s, colors, shapes, dic_percents, neighbors):
        idxs_c = []
        for i, label in enumerate(labels_c):
            app = True
            for color in colors:
                if (color not in label):
                    app = False
            if app == True:
               idxs_c.append(i)
        idxs_s = np.array([], dtype=int)
        for shape in shapes:
            idxs_s = np.append(idxs_s, np.where(labels_s == shape))
        idxs = np.intersect1d(idxs_c,idxs_s)
        idxs = np.array(idxs)
        ordena = np.empty((idxs.shape[0],))
        for m,n in enumerate(idxs):
            media = 0
            for c in colors:
                media += dic_percents[n][c]
            ordena[m] = media
        idxs = idxs[np.argsort(ordena)][::-1]

        return imgs[idxs],idxs


    def Kmean_statistics(kmeans, kmax,ground):
        stats = []
        fcs = np.array([])
        for k in range(2, kmax + 1):
            stat = np.empty(7, dtype="U30")
            kmeans.K = k
            t1 = t.time()
            kmeans.fit()
            temps = t.time() - t1
            if kmeans.options['fitting'] == 'WCD':
                heur = kmeans.withinClassDistance()
                stat[0] = k
                stat[1] = temps
                stat[2] = kmeans.num_iter
                stat[3] = heur
                stat[4] = '%DEC'
                if k == 2:
                    stat[5] = "No es possible trobar-lo!"
                    #stats = np.append(stats, stat)
                else:
                    dec_pctg = 100 * (1 - (heur / heur_aux))
                    stat[5] = (-dec_pctg / 100 + 1) * 100
                    #stats = np.append(stats, stat)
                heur_aux = heur
            elif kmeans.options['fitting'] == 'BSS':
                heur = kmeans.bSS_tSS()
                stat[0] = k
                stat[1] = temps
                stat[2] = kmeans.num_iter
                stat[3] = heur
                stat[4] = "diferencia amb l'anterior"
                if k == 2:
                    stat[5] = "No es possible trobar-lo!"
                    #stats = np.append(stats, stat)
                else:
                    dec_pctg = heur - heur_aux
                    stat[5] = dec_pctg
                    #stats = np.append(stats, stat)
                heur_aux = heur
            elif kmeans.options['fitting'] == 'FC':
                heur = kmeans.fisherCoeficient()
                stat[0] = k
                stat[1] = temps
                stat[2] = kmeans.num_iter
                stat[3] = heur
                fcs = np.append(fcs, heur)
                stat[4] = 'millor k fins ara'
                stat[5] = np.argmax(fcs) + 2
            stat[6] = Get_color_accuracy([get_colors(km.centroids)],[ground])
            stats = np.append(stats, stat)
        stats = [stats[i:i + 7] for i in range(0, len(stats), 7)]
        for stat in stats:
            print(
                "k = {}, temps per convergir = {}, nombre d'iteracions = {}, accuracy = {}, {} = {}, {} = {}".format(stat[0], stat[1],
                                                                                                      stat[2], stat[6],
                                                                                                      kmeans.options[
                                                                                                          'fitting'],
                                                                                                      stat[3], stat[4],
                                                                                                      stat[5]))
        return stats

    def Get_shape_accuracy(knn_labels, ground_truth):
        n_coincident = np.sum(knn_labels == ground_truth)
        return (n_coincident / ground_truth.shape[0]) * 100


    def Get_color_accuracy(kmeans_labels, ground_truth):
        accuracy = 0
        for label, truth in zip(kmeans_labels, ground_truth):
            accuracy += len(np.intersect1d(label, truth)) / (max(len(np.unique(label)), len(truth)))
        return (accuracy / len(kmeans_labels)) * 100

    '''
    #Demostració Resultats Retrieval by Color:
    colors = []
    color_percent = []
    #Fem predict de les imatges del reduced dataset
    for im in test_imgs:
        km = KMeans(im, 5)
        #km.find_bestK(10)
        km.fit()
        colors.append(get_colors(km.centroids))
    #Creem els percentatges de cada color
    for label in colors:
        col, ocurrencias = np.unique(label,return_counts=True)
        p = (ocurrencias/len(label))*100
        dict_per = dict(zip(col,p))
        color_percent.append(dict_per)
    #Cridem al retrieval i veiem per pantalla el resultat.
    cerca = ['Red']
    titol = 'Search clothes that contain/are '+str(cerca)
    visualize_retrieval(Retrieval_by_color(test_imgs, colors, cerca, np.array(color_percent)), 20, title=titol)
    '''

    '''
    #Demostració Resultats Retrieval by Shape
    knn = KNN(train_imgs_grey,train_class_labels)
    shapes = knn.predict(test_imgs_grey,4)
    neigh = knn.neighbors
    cerca = ['Heels','Shorts']
    titol = 'Search clothes that are ' + str(cerca)
    visualize_retrieval(Retrieval_by_shape(test_imgs, shapes, cerca,neigh), 20, title=titol)
    '''

    '''
    #Demostració de Retrieval combined
    colors = []
    color_percent = []
    for im in test_imgs:
        km = KMeans(im, 5)
        km.fit()
        colors.append(get_colors(km.centroids))
    for label in colors:
        col, ocurrencias = np.unique(label,return_counts=True)
        p = (ocurrencias/len(label))*100
        dict_per = dict(zip(col,p))
        color_percent.append(dict_per)
    knn = KNN(train_imgs_grey, train_class_labels)
    shapes = knn.predict(test_imgs_grey, 4)
    neigh = knn.neighbors

    cerca_c = ['Red', 'Black']
    cerca_s = ['Handbags','Dresses']
    titol = 'Search clothes that are the color '+str(cerca_c)+' and have shape '+str(cerca_s)
    ig,idx = Retrieval_combined(test_imgs, colors,shapes, cerca_c, cerca_s,np.array(color_percent),neigh)
    visualize_retrieval(ig, 20, title=titol)
    '''

    '''
    #Demostració del kmeans
    fits = ['WCD','BSS','FC']
    for fit in fits:
        for i, im in enumerate(cropped_images[:5]):
            km = KMeans(im,1)
            km.options['fitting'] = fit
            stats = Kmean_statistics(km,10,color_labels[i])
            if i == 0:
                x = [int(stat[0]) for stat in stats]
                imp = [1,2,6,3,5]
                names = ['temps','num_iteracions','exactitud',fit]
                for m, j in enumerate(imp):
                    if j==5 and fit!='FC':
                        stats[0][j] = 0
                    y = [float(stat[j]) for stat in stats]
                    plt.plot(x, y, marker='o', linestyle='-',color='red')
                    plt.xlabel('K')
                    if m == 4:
                        if fit == 'WCD':
                            plt.ylabel('%DEC')
                            plt.title('Evolució del %DEC')
                        elif fit == 'BSS':
                            plt.ylabel('tasa de canvi')
                            plt.title('Evolució de la tasa de canvi')
                        else:
                            plt.ylabel('millor k')
                            plt.title('Evolució de la millor k fins al moment')
                    else:
                        plt.ylabel(names[m])
                        plt.title('Evolució del '+names[m])
                    plt.show()
    '''

    '''
    #Demostració accuracy shape
    knn = KNN(train_imgs_grey,train_class_labels)
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[:10])
    print(test_class_labels[:10])
    print(str(Get_shape_accuracy(shapes,test_class_labels))+'%')
    '''

    '''
    #Demostració accuracy color
    colors = []
    for im in imgs:
        km = KMeans(im, 5)
        km.find_bestK(10)
        km.fit()
        colors.append(get_colors(km.centroids))

    print(str(Get_color_accuracy(colors,color_labels))+'%')
    print(['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue'],['Blue', 'Grey'])
    print(str(Get_color_accuracy([['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue']],[['Blue', 'Grey']]))+'%')
    print(['Green', 'Green', 'Blue', 'Green'],['Yellow', 'Green', 'Blue', 'White'])
    print(str(Get_color_accuracy([['Green', 'Green', 'Blue', 'Green']],[['Yellow', 'Green', 'Blue', 'White']]))+'%')
    print(['White', 'White', 'Grey', 'Black', 'Black', 'Black', 'Grey', 'Grey'],['Black', 'Grey'])
    print(str(Get_color_accuracy([['White', 'White', 'Grey', 'Black', 'Black', 'Black', 'Grey', 'Grey']],[['Black', 'Grey']]))+'%')
    '''

    '''
    #Demostració inicialització de centroides:
    ll = []
    for inici in ['first', 'random', 'equal_segments']:
        l = []
        for i, im in enumerate(imgs[:5]):
            km = KMeans(im,4)
            km.options['km_init'] = inici
            s = Kmean_statistics(km,10,color_labels[i])
            statforini = []
            for stat in s:
                statforini.append(stat[0])
                statforini.append(stat[1])
                statforini.append(stat[2])
            l.append(statforini)
        ll.append(l)

    sum_temps = []
    idx = [i for i in range(1,27,3)]
    for stat in ll:
        o = []
        for i in idx:
            ksum = 0
            itsum = 0
            for imag in stat:
                ksum += float(imag[i])
                itsum += float(imag[i+1])
            o.append(ksum/5)
            o.append(itsum/5)
        sum_temps.append(o)
    x = [k for k in range(2,11)]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for mid, ini in zip(sum_temps,['first', 'random', 'equal_segments']):
        y1 = [m for j, m in enumerate(mid) if j%2==0]
        y2 = [m for j, m in enumerate(mid) if j%2!=0]
        ax1.plot(x,y1,label=ini)
        ax2.plot(x,y2,label=ini)
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('K')
        ax2.set_xlabel('K')
        ax1.set_ylabel('Temps')
        ax2.set_ylabel("Número d'iteracions")
    plt.show()
    '''

    '''
    ka = []
    temps = []
    exac = []
    fits = ['WCD', 'BSS', 'FC']
    for fit in fits:
        kas = 0
        te = 0
        ex = 0
        for i, im in enumerate(cropped_images):
            km = KMeans(im,1)
            km.options['fitting'] = fit
            t1 = t.time()
            km.find_bestK(10)
            t2 = t.time()-t1
            km.fit()
            colors = get_colors(km.centroids)
            exactitud = (Get_color_accuracy([colors],[color_labels[i]]))
            #print("Per l'heurística",fit,"obtenim k="+str(km.K)+", tardem "+str(t2)[:5]+"s i obtenim una exactitud de "+str(exactitud)+'%')
            kas+=km.K
            te+=t2
            ex+=exactitud
        ka.append(kas/len(cropped_images))
        temps.append(te/len(cropped_images))
        exac.append(ex/len(cropped_images))

    for i,fit in enumerate(fits):
        print("Per l'heurística "+str(fit)+", de mitja, obtenim k="+str(int(round(ka[i],0)))+", tardem "+str(temps[i])[:5]+"s i obtenim una exactitud de "+str(exac[i])[:5]+'%')
    '''

    '''
    for k in [5,4,7]:
        for im in imgs[:4]:
            km = KMeans(im,k)
            km.fit()
            visualize_k_means(km,(80,60,3))
    '''

    '''
    ka = []
    temps = []
    exac = []
    fits = ['WCD', 'BSS', 'FC']
    centro = ['first', 'random', 'equal_segments']
    for fit in fits:
        kas = []
        te = []
        ex = []
        for c in centro:
            kkk = 0
            tip = 0
            exex = 0
            for i, im in enumerate(cropped_images[:80]):
                km = KMeans(im,1)
                km.options['fitting'] = fit
                km.options['km_init'] = c
                t1 = t.time()
                km.find_bestK(10)
                t2 = t.time()-t1
                km.fit()
                colors = get_colors(km.centroids)
                exactitud = (Get_color_accuracy([colors],[color_labels[i]]))
                kkk+=km.K
                tip+=t2
                exex+=exactitud
            kas.append(kkk/80)
            te.append(tip/80)
            ex.append(exex/80)
        ka.append(kas)
        temps.append(te)
        exac.append(ex)

    for i,fit in enumerate(fits):
        for j,c in enumerate(centro):
            print("Per l'heurística "+str(fit)+" i iniciant els centroides de manera "+str(c)+", de mitja, obtenim k="+str(int(round(ka[i][j],0)))+", tardem "+str(temps[i][j])[:6]+"s i obtenim una exactitud de "+str(exac[i][j])[:5]+'%')

    '''
    '''
    tempo = []
    exacs = []
    sizes = [i for i in range(80,20,-1) if (i*0.75).is_integer()]
    for s in sizes:
        train = []
        for im in train_imgs_grey:
            p = np.copy(im)
            a = np.resize(p, (s, int(s*0.75)))
            train.append(a)
        train = np.array(train)

        test = []
        for im in test_imgs_grey:
            p = np.copy(im)
            a = np.resize(p, (s, int(s*0.75)))
            test.append(a)
        test = np.array(test)
        t1 = t.time()
        knn = KNN(train, train_class_labels)
        shapes = knn.predict(test, 4)
        t2 = t.time() - t1
        exac = (Get_shape_accuracy(shapes,test_class_labels))
        temps = t2
        exacs.append(exac)
        print("Per imatges de tamany ("+str(s)+","+str(int(s*0.75))+") tardem "+str(temps)[:5]+"s i obtenim una exactitud de "+str(exac)[:4]+"%")
        tempo.append(temps)


    tempo = np.array(tempo)
    exacs = np.array(exacs)
    coef = []
    for j, (t,e) in enumerate(zip(tempo,exacs)):
        if j!=0:
            coef.append((exacs[0]-e)/(tempo[0]-t))
        else:
            coef.append(10)
    coef = np.array(coef)
    print(coef)
    print(sizes[np.argmin(coef)])
    '''

    '''
    for k in range(2,10):
        train = []
        for im in train_imgs_grey:
            p = np.copy(im)
            a = np.resize(p, (64,48))
            train.append(a)
        train = np.array(train)

        test = []
        for im in test_imgs_grey:
            p = np.copy(im)
            a = np.resize(p, (64,48))
            test.append(a)

        test = np.array(test)
        t1 = t.time()
        knn = KNN(train, train_class_labels)
        shapes = knn.predict(test, k)
        t2 = t.time() - t1
        exac = (Get_shape_accuracy(shapes,test_class_labels))
        temps = t2
        print("Per a "+str(k)+" veïns tardem "+str(temps)[:5]+"s i obtenim una exactitud de "+str(exac)[:4]+"%")
    '''

    '''
    fit = ['WCD','BSS']
    inici = ['first','random']
    h = [80, 64]
    xd = ['sense','amb']

    for f, i, h, x in zip(fit,inici,h,xd):
        t1 = t.time()
        colors = []
        for im in test_imgs:
            km = KMeans(im,1)
            km.options['fitting'] = f
            km.options['km_init'] = i
            km.find_bestK(10)
            km.fit()
            co = get_colors(km.centroids)
            colors.append(co)
        if h!=80:
            train = []
            for im in train_imgs_grey:
                p = np.copy(im)
                a = np.resize(p, (64,48))
                train.append(a)
            train = np.array(train)

            test = []
            for im in test_imgs_grey:
                p = np.copy(im)
                a = np.resize(p, (64,48))
                test.append(a)
            test = np.array(test)
        else:
            train = train_imgs_grey
            test = test_imgs_grey

        knn = KNN(train, train_class_labels)
        shapes = knn.predict(test, 4)
        t2 = t.time() - t1
        exac_c = Get_color_accuracy(colors,test_color_labels)
        exac_s = Get_shape_accuracy(shapes,test_class_labels)
        print("Execució "+x+" millores: Temps = "+str(t2)[:6]+"s Exactitud = "+str((exac_c+exac_s)/2)[:5]+"%")
    '''

    '''
    km = KMeans(cropped_images[0],4)
    km.fit()
    Plot3DCloud(km)
    plt.show()
    '''


