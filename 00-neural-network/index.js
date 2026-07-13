import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  //First layer
  // input of 7 positions (age, 3 colors and 3 locations)

  // 80 neurons = I put all this because we have little training data
  // the more neurons the more complexity

  // ReLU acts as a filter:
  // It's like it only lets interesting data continue in the network
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // output: 3 neurons (because we have three categories)
  // activation: softmax normalizes the output into probabilities
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Compiling the model
  // optimizer adam -> adaptive moment estimation
  // is a modern personal trainer for neural networks
  // that adjusts weights efficiently and intelligently
  // it will learn from history of hits and errors
  //
  // Loss: categoricalCrossentropy -> compares what the model "thinks" (scores for each category)
  // with the correct answer
  //
  // the more distant the model's prediction from the correct answer
  // the greater the error (loss)
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  //model training
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    // callbacks: {
    //   onEpochEnd: (epoch, log) =>
    //     console.debug(`Epoch: ${epoch} | loss: ${log.loss}`),
    // },
  });

  return model;
}

async function predict(model, person) {
  //convert js array to tensor2d
  const tfInput = tf.tensor2d(person);

  // Make prediction according to the trained model
  const pred = model.predict(tfInput);
  const predArray = await pred.array();

  return predArray[0].map((prob, index) => ({ prob, index }));
}

//Neural network training data

// Example people for training (each person with age, color and location)
// const people = [
//     { name: "Erick", age: 30, color: "blue", location: "São Paulo" },
//     { name: "Ana", age: 25, color: "red", location: "Rio" },
//     { name: "Carlos", age: 40, color: "green", location: "Curitiba" }
// ];

// Input vectors with already normalized and one-hot encoded values
// Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
// const tensorPeople = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// We use only numeric data, since neural networks only understand numbers.
// normalizedPersonTensor corresponds to the model's input dataset.
const trainingData = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Category labels to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelNames = ["premium", "medium", "basic"]; // Labels order
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// We create input tensors (xs) and output tensors (ys) to train the model
const inputXs = tf.tensor2d(trainingData);
const outputYs = tf.tensor2d(tensorLabels);

// inputXs.print();
// outputYs.print();

const model = await trainModel(inputXs, outputYs);

// const person = {
//   name: "Zé",
//   age: 28,
//   color: "green",
//   location: "Curitiba",
// };

// Normalizing the new person's age using the same training pattern
// Example: age_min = 25, age_max = 40, so it will be (28 - 25) / (40 - 25) = 0.2
const normalizedPersonTensor = [[0.2, 1, 0, 0, 1, 0, 0]];

const predictions = await predict(model, normalizedPersonTensor);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelNames[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");

console.log(results);
