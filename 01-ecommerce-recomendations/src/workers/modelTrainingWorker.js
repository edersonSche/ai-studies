import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

// Normalize data between 0 and 1 so the training can better understand the data being passed
const normalize = (value, min, max) => (value - min) / (max - min || 1);

function makeContext(products, users) {
  const ages = users.map((u) => u.age);
  const prices = products.map((p) => p.price);
  const colors = [...new Set(products.map((p) => p.color))];
  const categories = [...new Set(products.map((p) => p.category))];

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  const colorsIndex = Object.fromEntries(
    colors.map((item, index) => [item, index]),
  );

  const categoriesIndex = Object.fromEntries(
    categories.map((item, index) => [item, index]),
  );

  // Compute the average age of buyers per product (helps with personalization)
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((p) => {
      ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
      ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
    });
  });

  const productsAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    products,
    users,
    colorsIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    productsAvgAgeNorm,
    //price + age + categories + colors
    dimentions: 2 + categories.length + colors.length,
  };
}

const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  const age = tf.tensor1d([
    context.productsAvgAgeNorm[product.name] ?? 0.5 * WEIGHTS.age,
  ]);

  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    context.colorsIndex[product.color],
    context.numColors,
    WEIGHTS.color,
  );

  return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, context)))
      .mean(0)
      .reshape([1, context.dimentions]);
  }

  return tf
    .contatId([
      tf.zeros([1]), // price is ignored
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.numCategories]), // category ignored
      tf.zeros([context.numColors]), // color ignored
    ])
    .reshape([1, context.dimentions]);
}

function createTrainingData(context) {
  const inputs = [];
  const labels = [];

  context.users
    .filter((u) => u.purchases.length)
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();
      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();
        const label = user.purchases.some(
          (purchase) => purchase.name === product.name,
        )
          ? 1
          : 0;

        // combine user + product
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimention: context.dimentions * 2,
  };
}

async function configAndTrainNeuralNetwork(trainingData) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [trainingData.inputDimention],
      units: 128,
      action: "relu",
    }),
  );
  // with the data from the first layer, pass through the filter again
  model.add(tf.layers.dense({ units: 64, action: "relu" }));
  // with the data from the second layer, pass through the filter again
  model.add(tf.layers.dense({ units: 32, action: "relu" }));
  // output layer
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainingData.xs, trainingData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });

  const products = await (await fetch("/data/products.json")).json();
  const context = makeContext(products, users);
  context.productVectors = products.map((product) => ({
    name: product.name,
    meta: { ...product },
    vector: encodeProduct(product, context).dataSync(),
  }));

  _globalCtx = context;

  const trainingData = createTrainingData(context);
  _model = await configAndTrainNeuralNetwork(trainingData);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}

function recommend(user, ctx) {
  if (!_model) return;

  const context = _globalCtx;

  console.log("will recommend for user:", user);
  const userVector = encodeUser(user, context).dataSync();
  const inputs = context.productsVector.map(({ vector }) => [
    ...userVector,
    ...vector,
  ]);
  const inputTensor = tf.tensor2d(inputs);
  const predictions = _model.predict(inputTensor);

  const scores = predictions.dataSync();
  const recommendations = context.productsVector
    .map((item, index) => {
      return {
        ...item.meta,
        name: item.name,
        score: scores[index],
      };
    })
    .sort((a, b) => b.score - a.score);

  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
