from __future__ import print_function, division

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import empty
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import re

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.mail_rows = 8
        self.mail_cols = 8
        self.mail_shape = (self.mail_rows, self.mail_cols, 1)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the discriminator2
        self.discriminator2 = self.build_discriminator()
        self.discriminator2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates mails
        z = Input(shape=(self.latent_dim,))
        mail = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(mail)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.mail_shape), activation='tanh'))
        model.add(Reshape(self.mail_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        mail = model(noise)

        return Model(noise, mail)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.mail_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        mail = Input(shape=self.mail_shape)
        validity = model(mail)

        return Model(mail, validity)

    def results(self, pred, actual):
        results = confusion_matrix(actual, pred)
        print('Confusion Matrix :')
        print(results)
        print ('Accuracy Score :',accuracy_score(actual, pred))
        print ('Report : ')
        print(classification_report(actual, pred))
        print()

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Test data
        disc_loss = [0] * epochs
        gen_loss = [0] * epochs

        disc_acc= [0] * epochs
        gen_acc= [0] * epochs

        x = np.loadtxt('../spambase/spambase.data', delimiter=',')
        y = x[:, 57]

        data_max = np.zeros(58)
        data_min= np.zeros(58)
        for n in range(len(x[0])):
            data_max[n] = max(x[:, n])
            data_min[n] = min(x[:, n])

        data = empty([x.shape[0], 58])
        for r in range(len(x)):
            for c in range(len(x[0])):
                if(c != 57):
                    data[r, c] = ((x[r, c] - data_min[c]) / (data_max[c] - data_min[c]))
                else:
                    data[r, c] = x[r, c]

        X_train, X_t, Y_train, Y_t = train_test_split(data, y, test_size = 0.3, random_state = 0)

        ts = empty([X_t.shape[0], 65])
        for n in range(len(X_t)):
            ts[n] = np.append(X_t[n], [0, 0, 0, 0, 0, 0, 0], axis=0)

        X_test = np.asarray(ts)
        X_test[:, 57] = 0
        Y_test = np.asarray(Y_t)

        X_test = X_test[:, : 64]
        X_test = X_test.reshape(X_test.shape[0], 8, 8)
        # Error could be here
        Y_test = Y_test.reshape(Y_test.shape[0], 1)
        #X_test = X_test / 127.5 - 1
        X_test = np.expand_dims(X_test, axis=3)

        spam = []
        email = []

        tr = empty([X_train.shape[0], 65])
        for n in range(len(X_train)):
            tr[n] = np.append(X_train[n], [0, 0, 0, 0, 0, 0, 0], axis=0)
            if tr[n][57]== 1:
                tr[n][57] = 0
                spam.append(tr[n])
            else:
                tr[n][57] = 0
                email.append(tr[n])

        X_train_spam = np.asarray(spam)
        X_train_email = np.asarray(email)

        X_train_spam = X_train_spam[:, : 64]  # verify again the actual value that it should have
        # change the 8's by self.mail_rows
        X_train_spam = X_train_spam.reshape(X_train_spam.shape[0], 8, 8)

        X_train_email = X_train_email[:, : 64]  # verify again the actual value that it should have

        # change the 8's by self.mail_rows
        X_train_email = X_train_email.reshape(X_train_email.shape[0], 8, 8)

        X_train_spam = np.expand_dims(X_train_spam, axis=3)

        X_train_email = np.expand_dims(X_train_email, axis=3)

        # Adversarial ground truths
        valid_spam = np.ones((batch_size, 1))
        valid_email = np.zeros((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            # Train the discriminator
            # Select a random batch of images
            idx_spam = np.random.randint(0, X_train_spam.shape[0], batch_size)
            idx_email = np.random.randint(0, X_train_email.shape[0], batch_size)

            mails_spam = X_train_spam[idx_spam]
            mails_spam[:, 7, 1:] = 0

            mails_email = X_train_email[idx_email]
            mails_email[:, 7, 1:] = 0

            idx_test = np.random.randint(0, X_test.shape[0], batch_size)
            mails_test = X_test[idx_test]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_mails = self.generator.predict(noise)
            gen_mails[:, 7, 1:] = 0

            # Train the normal model
            n_loss = self.discriminator2.train_on_batch(mails_spam, valid_spam)
            n_loss_fake = self.discriminator2.train_on_batch(mails_email, valid_email)
            n_loss = 0.5 * np.add(n_loss, n_loss_fake)

            # Train the discriminator
            d_loss_real_spam = self.discriminator.train_on_batch(mails_spam, valid_spam)
            d_loss_real_email = self.discriminator.train_on_batch(mails_email, valid_email)
            d_loss_fake = self.discriminator.train_on_batch(gen_mails, fake)

            d_tot = np.add(d_loss_real_spam, d_loss_fake)
            d_loss = 0.33 * np.add(d_tot, d_loss_real_email)


            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid_email)

            disc_loss[epoch] = d_loss[0]
            gen_loss[epoch] = g_loss

            disc_acc[epoch] = 100 * d_loss[1]
            gen_acc[epoch] = 100 *  n_loss[1]

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, nacc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], 100*n_loss[1], g_loss))

        # Generate a batch of new images
        gen = 10000
        noise = np.random.normal(0, 1, (gen, self.latent_dim))
        new_mails = self.generator.predict(noise)
        new_mails[:, 7, 1:] = 0
        Y_noise = np.ones((gen, 1))

        predicted = np.round(self.discriminator.predict_on_batch(X_test))
        actual = Y_test
        print('This are predictions on the test dataset using discriminant: \n')
        self.results(predicted, actual)

        predicted2 = np.concatenate((np.round(self.discriminator.predict_on_batch(X_train_spam)), np.round(self.discriminator.predict_on_batch(X_train_email))))
        actual2 = np.concatenate((np.ones((1254, 1)), np.zeros((1966, 1))))
        print(predicted2.shape)
        print(np.round(self.discriminator.predict_on_batch(X_train_spam)).shape)
        print(np.round(self.discriminator.predict_on_batch(X_train_email)).shape)
        print('This are predictions on the training dataset using discriminant: \n')
        self.results(predicted2, actual2)

        predicted_n = np.round(self.discriminator.predict_on_batch(new_mails))
        print('This are predictions on the generated dataset using discriminant: \n')
        self.results(predicted_n, Y_noise)

        predicted = np.round(self.discriminator2.predict_on_batch(X_test))
        print('This are predictions on the test dataset using a NN: \n')
        self.results(predicted, actual)

        predicted2 = np.concatenate((np.round(self.discriminator2.predict_on_batch(X_train_spam)), np.round(self.discriminator2.predict_on_batch(X_train_email))))
        print('This are predictions on the training dataset using a NN: \n')
        self.results(predicted2, actual2)

        predicted_n =  np.round(self.discriminator2.predict_on_batch(new_mails))
        print('This are predictions on the generated dataset using a NN: \n')
        self.results(predicted_n, Y_noise)

        x = np.arange(0, epochs, 50)

        plt.subplot(1, 2, 1)
        m = np.asarray(disc_loss)
        y = m[x]
        plt.plot(x, y, label = 'Discriminant Loss')

        m = np.asarray(gen_loss)
        y = m[x]
        plt.plot(x, y, label = 'Generator Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        m = np.asarray(disc_acc)
        y = m[x]
        plt.plot(x, y, label = 'Discriminant Acc')

        m = np.asarray(gen_acc)
        y = m[x]
        plt.plot(x, y, label = 'Normal Model Acc')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=500, batch_size=200, sample_interval=800)

