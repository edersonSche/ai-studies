# E-commerce Recommendation System

A browser-based e-commerce application that uses a **TensorFlow.js neural network** running inside a **Web Worker** to train on user purchase history and generate personalized product recommendations in real time.

## Tech Stack

- Vanilla ES Modules (browser, no bundler)
- TensorFlow.js 4.22.0 (CDN, Web Worker)
- tfjs-vis 1.5.1 (CDN, main thread)
- Bootstrap 5.3 + Bootstrap Icons
- browser-sync (dev server)

## Project Structure

```
01-ecommerce-recomendations/
  index.html                          # SPA shell (3-column layout)
  style.css                           # Application styles
  package.json                        # Manifest (browser-sync dev server)
  data/
    products.json                     # 10-product catalog
    users.json                        # 5 users with purchase history
  src/
    index.js                          # Entry point / bootstrap
    controller/
      UserController.js               # User selection, purchase history CRUD
      ProductController.js            # Product grid, buy actions, recommendation rendering
      ModelTrainingController.js      # Train/recommend buttons, progress UI
      WorkerController.js             # Main thread <-> Web Worker bridge
      TFVisorController.js            # tfjs-vis visualization controller
    view/
      View.js                         # Base view (template loading, {{placeholder}} substitution)
      UserView.js                     # User dropdown, age display, past purchases list
      ProductView.js                  # Product card grid, buy buttons
      ModelTrainingView.js            # Train/recommend buttons, progress bar, all-users panel
      TFVisorView.js                  # Live accuracy/loss charts during training
      templates/
        product-card.html             # Product card template
        past-purchase.html            # Past purchase item template
    service/
      UserService.js                  # User CRUD via sessionStorage
      ProductService.js               # Product catalog fetch
    events/
      constants.js                    # Event name constants (DOM + Worker)
      events.js                       # Event bus (CustomEvent on document)
    workers/
      modelTrainingWorker.js          # TF.js model training + recommendation engine
```

## Architecture

### MVC + Event Bus + Web Worker

```
                    +----------------+
                    |   Events.js    |
                    | (CustomEvent   |
                    |  bus on        |
                    |  document)     |
                    +-------+--------+
                            |
         +------------------+------------------+
         |                  |                  |
  +------v------+  +-------v-------+  +-------v-------+
  | UserController | | ModelController | | ProductController |
  +------+-------+  +-------+-------+  +-------+-------+
         |                  |                  |
  +------v------+  +-------v-------+  +-------v-------+
  | UserView    |  | ModelView     |  | ProductView   |
  | (select,    |  | (buttons,     |  | (card grid,   |
  |  age, past  |  |  progress,    |  |  buy buttons) |
  |  purchases) |  |  all-users)   |  +---------------+
  +-------------+  +-------+-------+
                            |
                   +--------v--------+
                   | WorkerController | <--> Web Worker
                   +-----------------+       |
                                   +---------v-----------+
                                   | modelTrainingWorker |
                                   | (TF.js training,    |
                                   |  recommendations)   |
                                   +---------------------+

  TFVisorController --> TFVisorView (tfjs-vis live charts)
```

All inter-component communication flows through the **event bus** (`Events.js`) using DOM `CustomEvent` on `document`. The `WorkerController` bridges between DOM events and the Web Worker's `postMessage` protocol.

## Features

- **User profile selection** with age display and past purchase history
- **Buy Now** functionality with green flash animation on purchase
- **Click-to-remove** past purchases with fade-out animation
- **Automatic model training** on application startup
- **Live training visualization** via tfjs-vis (accuracy + loss charts)
- **Personalized recommendations** generated when a user is selected
- **Retrain** button to retrain the model with updated purchase data
- **Collapsible all-users purchase summary** panel
- **Cold-start support** for new users with no purchase history

## How the ML Model Works

### Neural Network Architecture

```
Input (28 dims) --> Dense(128, relu) --> Dense(64, relu) --> Dense(32, relu) --> Dense(1, sigmoid)
```

- **Optimizer:** Adam (learning rate 0.01)
- **Loss:** Binary cross-entropy
- **Epochs:** 100 | **Batch size:** 32

### Feature Encoding

Each product and user is encoded into a fixed-size vector:

| Feature | Dimension | Weight | Encoding |
|---------|-----------|--------|----------|
| Price | 1 | 0.2 | Min-max normalized |
| Avg buyer age | 1 | 0.1 | Min-max normalized |
| Category | numCategories | 0.4 | One-hot encoded |
| Color | numColors | 0.3 | One-hot encoded |

**Product encoding:** Direct feature vector.

**User encoding (with purchases):** Mean of all purchased product vectors (represents the user's "taste profile").

**User encoding (no purchases):** Only age is populated, everything else is zero (cold-start).

### Training Data Generation

For each user-product pair:
- **Input:** Concatenation of `[userVector || productVector]` (28 + 28 = 56 dimensions)
- **Label:** `1` if the user purchased that product, `0` otherwise

### Recommendation Flow

1. User is selected from dropdown
2. User vector is encoded
3. All products are paired with the user vector
4. Model predicts a score for each pair
5. Products are sorted by score descending (highest = most recommended)

## Setup and Run

1. Install dependencies:
```
npm install
```

2. Start the application:
```
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Data

### Products (10 items)

| ID | Name | Category | Price | Color |
|----|------|----------|-------|-------|
| 1 | Wireless Headphones | electronics | $129.99 | black |
| 2 | Smart Watch | electronics | $199.99 | silver |
| 3 | Bluetooth Speaker | electronics | $89.99 | blue |
| 4 | Printed T-Shirt | clothing | $49.99 | white |
| 5 | Slim Fit Jeans | clothing | $99.99 | blue |
| 6 | Sports Sneakers | footwear | $149.99 | red |
| 7 | Casual Sandal | footwear | $69.99 | beige |
| 8 | Stylish Cap | accessories | $39.99 | black |
| 9 | Executive Backpack | accessories | $159.99 | gray |
| 10 | Sunglasses | accessories | $89.99 | brown |

### Users (5 seed users + 1 test user)

| User | Age | Purchases |
|------|-----|-----------|
| Ana Lima | 25 | Wireless Headphones, Smart Watch |
| Bruno Ferreira | 27 | Wireless Headphones, Bluetooth Speaker |
| Camila Souza | 30 | Printed T-Shirt, Slim Fit Jeans |
| Diego Almeida | 22 | Smart Watch, Bluetooth Speaker, Sports Sneakers |
| Eduarda Nunes | 28 | Wireless Headphones, Sports Sneakers, Slim Fit Jeans |
| John Smith | 30 | *(none - cold-start test user)* |

## Event System

### DOM Events (CustomEvent on document)

| Event | Trigger | Action |
|-------|---------|--------|
| `user:selected` | User selected from dropdown | Enables buy buttons, triggers recommendations |
| `users:updated` | Any user data change | Refreshes all-users purchase display |
| `purchase:added` | Buy Now clicked | Adds purchase to user, updates UI |
| `purchase:remove` | Past purchase clicked | Removes purchase from user, fades out element |
| `training:train` | Train Model clicked | Sends users to worker for training |
| `training:complete` | Worker finishes training | Enables Run Recommendation button |
| `model:progress-update` | Worker training progress | Updates progress bar/spinner |
| `recommendations:ready` | Worker finishes recommendations | Re-renders product grid with scores |

### Web Worker Messages

**Main -> Worker:**
```javascript
{ action: 'train:model', users: User[] }
{ action: 'recommend', user: User }
```

**Worker -> Main:**
```javascript
{ type: 'progress:update', progress: { progress: 50 | 100 } }
{ type: 'training:complete' }
{ type: 'training:log', epoch, loss, accuracy }
{ type: 'recommend', user, recommendations }
```
