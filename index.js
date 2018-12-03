// Texture for the particle
let particle_texture = null;

// Variable holding the particle system
let ps = null;

// Variables for the wind
let dx;
let dy;
let wind;

// Variables describing the spatial extent of the fire
let min_y;
let min_x;
let max_x;

// Normalized versions of the variables above
let min_y_n;
let min_x_n;
let max_x_n;

// Dimensions for the canvas
const page_width = 640;
const page_height = 360;

// Velocity of the ball
let vx = 0;
let vy = 0;

// New velocity of the ball
let new_vx = 0;
let new_vy = 0;
 
// Acceleration of the ball
let ax = 0;
let ay = 9.18;
 
// Damping factor for velocity
let vMultiplier = 0.007;
// Damping factor for bumping
let bMultiplier = 0.6;

// Deep Q Learning Variables
let action;
let survive;
let state_size = 3;
let action_size = 16;
let memory = [];
// Exploration rate
let epsilon = 1.0
// Factor by which the number of explorations decreases over time
let epsilon_decay = 0.95; 
// Lowest value of epsilon allowed
let epsilon_min = 0.01;
// Learning rate for neural network
let learning_rate = 0.01; 
let batch_size = 5;
let reward = 1;
let pred_x = [];
let pred_y = 0;

// DOM variables
let slider1;
let slider2;
let button1;
let button2;
let whichDiv;
let div1;
let div2;
let div3;
let explanationDiv1;
let explanationDiv2;
let explanationDiv3;
let explanationDiv4;
let explanationDiv5;
let explanationDiv6;
let explanationDiv7;
let explanationDiv8;
let explanationDiv9;
let explanationDiv10;
let explanationDiv11;
let explanationDiv12;
let explanationDiv13;
let explanationDiv14;
let explanationDiv15;
let explanationDiv16;

// Variables detemrinig display of DOM variables
let button_press = 0;
let ai_ready = 0;

// Create a shallow multi-layer perceptron to act as the AI agent
// Input: Variables describing the spatial extent of the fire (a human would likewise be
// able to see where the fire is located on the screen).
// Output: One of sixteen possible configurations of the sliders, which determine
// the upward and rightward force with which to shoot the ball.
const model = tf.sequential();
model.add(tf.layers.dense({units: 24, activation: 'relu',kernelInitializer:'glorotUniform',inputShape: [3]})); 
model.add(tf.layers.dense({units: action_size, activation: 'sigmoid',kernelInitializer:'glorotUniform'})); 
model.compile({loss: 'binaryCrossentropy', optimizer: 'adam'});

/**
 * Adds the current state, action, and associated reward to the "memory" array using
 * the remember() function.
 */
function list_append() {
    // The "state" are the three variables that describe the fire's spatial extent. 
    // "min_y_n" is the (normalized) highest y value that the fire reaches
    // "min_x_n" is the (normalized) lowest x value that the fire reaches
    // "max_x_n" is the (normalized) highest x value that the fire reaches
    let state = [min_y_n,min_x_n,max_x_n];
    // The "survive" variable indicates whether the ball sucecssfully made it to the
    // other side (1) or not (0).
    // The "reward" is (1) for success and (0) for failure. In this way, the neural
    // network will be incentivized to learn an optimal configuration of the sliders
    // to successfully shoot the ball over the fire.
    if (survive==0) {
        reward = 0;
    }
    if (survive==1) {
        reward = 1;
    }
    remember(state,action,reward);
}

/**
 * The "memory" array allows for experience replay. Every grouping of state, action, and 
 * reward is added as an element to the array.
 * @param state Current State
 * @param action Chosen Action
 * @param reward Received Reward
 */
function remember(state,action,reward) {
    let temp = [state,action,reward];
    memory.push(temp);
}

/**
 * Determines what action to take based on the current state
 * @param state Current State
 */
function act(state) {
    // Create an array to hold a one-hot encoding of the desired action for the current
    // state. In other words, all elements of the array are equal to zero except one, 
    // the index of which indicates which action is desired.
    action = (arr=[]);
    action.length = action_size;
    let action_index;
    let act_values;
    action.fill(0);
    // The value of "epsilon" determines whether an exploration or exploitation 
    // strategy should be adopted. (If a random number between 0 and 1 is less than
    // the value of "epsilon", an exploration strategy is taken, i.e. a randomly
    // chosen action. On the other hand, if a random number between 0 and 1 is greater
    // than the value of "epsilon," an exploitation strategy is taken, i.e. an action
    // based on prior experience. "Epsilon" is initialized to 1 and decreased gradually
    // with every iteration so that the agent is encouraged to explore the state space
    // initially and over time rely more upon its learned experiences. 
    if (Math.random() <= epsilon) {
        // Exploration: The action taken is random
        action_index = Math.floor(Math.random() * (action_size));
    } else {
        // Exploitation: The neural network makes a prediction based on the current
        // state and outputs what it believes to be the action that will succeed with
        // the most probability
        act_values = model.predict(state).dataSync();
        act_values = Array.from(act_values);
        action_index = argMax(act_values);
    }
    action[action_index] = 1;
}

/**
 * Helper function to get a random group of elements from an array
 * @param arr Array
 * @param size Specifies how large the random group of elements should be
 */
function getRandomSubarray(arr, size) {
    let shuffled = arr.slice(0), i = arr.length, min = i - size, temp, index;
    while (i-- > min) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(min);
}

/**
 * Helper function to determine the index corresponding to the maximum of an array
 * @param array Array
 */
function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

// The following set of three functions allow for experience replay and the training
// of the neural network.
/**
 * High-level function that calls the middle-level function replay() and determines how
 * many mini-batches with which to train the neural network
 */
async function sup_replay() {
    // The number of mini-batches is determined by the formula below, which relies upon
    // the current value of "epsilon." The number of batches will increase as the 
    // value of "epsilon" decreases (encouraging more training when there is a higher
    // probability of the exploitation strategy).
    let num_batches = Math.floor((100-(epsilon*100))/8)+1;
    let num_batches_array = Array.from(new Array(num_batches), (x,i) => i);
    for (const batch in num_batches_array) {
        await replay();
    }
}

/**
 * Middle-level function that calls the low-level function sub_replay().
 * Creates mini-batches out of random groups of elements from the "memory" array and
 * passes each sample of the mini-batch to sub_replay().
 */
async function replay() {
    // Get random group of elements from "memory" to create a mini-batch
    let minibatch = getRandomSubarray(memory,batch_size)
    // Pass each sample of the mini-batch to sub_replay().
    for (const sample of minibatch) {
        await sub_replay(sample);
    }
    console.log('Model fit');
    // "Epsilon_min" is the lowest defined acceptable value of "Epsilon." 
    // Multiply "epsilon" by the "epsilon_decay" factor to decrease the value of 
    // "epsilon" over time. 
    if (epsilon > epsilon_min) {
        epsilon = epsilon*epsilon_decay;
    }
}

/**
 * Low-level function that executes training of the neural network on a sample-by-sample
 * basis.
 * @param sample Sample with which to train the neural network
 */
async function sub_replay(sample) {
    // Classically, Q-tables have as many rows as possible states and as many columns as 
    // possible actions, with the value of each cell being the maximum expected future
    // reward (Q-value) for the given state and action. As the agent performs actions
    // in the state space and receives rewards, these Q-values are updated using the
    // Bellman Equation. New Q-values are a function of the received reward for the 
    // chosen action and the maximum expected future reward given all possible actions
    // in the new state. 
    // The goal of the neural network is to replace the Q-table by approximating the 
    // Q-value of each action for each state. 
    // The input is an array of 3 elements: state, action, reward
    let target = sample[2];
    let state = sample[0];
    // Have the model predict the Q-values for each possible action given the state of
    // the sample. 
    // The only Q-value that needs to be updated is the one corresponding to the action
    // taken in this sample. Identify the index of this action using argMax() and 
    // replace the predicted Q-value with the received reward (1) or (0). This will 
    // encourage the neural network to reduce the loss between the reward and predicted
    // Q-value. 
    let action_argmax = argMax(sample[1]);
    let target_f = model.predict(tf.tensor(state).reshape([1,3])).dataSync();
    target_f[action_argmax] = target;
    console.log('Fitting model');
    await model.fit(tf.tensor(state).reshape([1,3]),tf.tensor(target_f).reshape([1,action_size]),{
        epochs: 1
    });
}

// This function applies for the AI player
// Convert the one-hot vector indicating the chosen action for the current state into
// a setting for the "Rightward Force" and "Upward Force" sliders: "new_vx" and
// "new_vy," respectively. Note that the action is translated into quantized levels
// of the sliders (for simplicity). 
/**
 * Convert chosen action into settings for the sliders
 * @param action Chosen action
 */
function onehot_to_action(action) {
    let output = argMax(action);
    new_vx = ((output%4)*0.25)+0.25;
    new_vx = 4000*new_vx; 
    new_vy = Math.floor(output/4)+0.25;
    new_vy = -2000*new_vy; 
}

/**
 * Set the x and y values of the ball's velocity vector to the desired new values
 */
function go() {
    vx = new_vx;
    vy = new_vy;
}

// This function applies for the Human player
// Convert the levels of the sliders set by the Human player to a one-hot encoding
// that can be given to the neural network for training. 
/**
 * Convert the settings of the sliders into a chosen action
 */
function action_to_onehot() {
    let temp_vx = new_vx/4000;
    let temp_vy = new_vy/-2000; 
    temp_vx = Math.floor(temp_vx*16)%4;
    temp_vy = Math.floor(Math.floor((temp_vy-0.001)*16)/4); 
    temp_vy = temp_vy*4;
    action_index = temp_vy+temp_vx;
    action = (arr=[]);
    action.length = action_size;
    action.fill(0);
    action[action_index] = 1;
}

/**
 * Normalize the values for "state" using the width and height of the canvas for
 * input into the neural network.
 */
function normalize_inputs() {
    min_y_n = min_y/page_height;
    min_x_n = min_x/page_width;
    max_x_n = max_x/page_width;
}

/**
 * Resets the ball back in its starting position and calls the top-level function
 * sup_replay() to train the neural network if enough elements exist in "memory." Also
 * randomly changes the direction of the fire for the next iteration of the game.
 */
async function reset() {
    x = 20;
    y = page_height;
    vx = 0;
    vy = 0;
    dx = random(-0.2,0.2);
    dy = random(0,-0.2);
    wind = createVector(dx,dy);
    // Train the neural network only if there are enough elements to form a complete
    // mini-batch.
    if (memory.length >= batch_size) {
        await sup_replay();
        // The "ai_ready" variable shows the "Go!" button. The "Go!" button is hidden
        // while the neural network is training and only appears once training is 
        // complete. This avoids calling sup_replay() while training is taking place.
        ai_ready = 1;
    }
}

/**
 * Callback for the "Go!" button (for when it is pressed)
 */
function go_button() {
    // A value of 1 for "button_press" indicates the AI player
    if (button_press == 1) {
        // Normalize the values corresponding to the current state
        normalize_inputs();
        // Convert the state into a tensor
        let state = tf.tensor([min_y_n,min_x_n,max_x_n]).reshape([1,3]);
        // Determine what action to take based on the state and current value of 
        // "epsilon."
        act(state);
        // Convert the chosen action into values of the sliders on the canvas
        onehot_to_action(action);
        slider1.value(new_vx);
        slider2.value(new_vy);
        // Set the sliders to the desired new values
        go();
        // If the "Rightward Force" is too low (i.e. less than 100), the ball will
        // take an undesirable amount of time to travel across the canvas. In this
        // case, call reset() after 2 seconds. 
        if (abs(new_vx) < 100) {
            setTimeout(reset,2000);
        }
        // Once training of the neural network is able to occur (i.e. when the "memory"
        // vector is long enough), automatically set "ai_ready" to 0 so that the "Go!"
        // button is hidden. It will be shown again once training is complete.
        // On the other hand, when the "memory" vector is too short, automatically 
        // set "ai_ready" to 1 so that the "Go!" button will be consistently shown.
        // (There is no need to wait for the model to train to show the button). 
        if (memory.length >= batch_size) {
            ai_ready = 0;
        } else if (memory.length < batch_size) {
            ai_ready = 1;
        }
    // A value of 0 for "button_press" indicates the Human player
    } else if (button_press == 0) {
        // Set "new_vx" and "new_vy" to the values the human player set on the sliders
        new_vx = slider1.value();
        new_vy = slider2.value();
        // Normalize the values corresponding to the current state
        normalize_inputs();
        // Convert the values set on the sliders to an action that can be used to 
        // train the neural network. 
        action_to_onehot();
        // Set the sliders to the desired new values
        go();
        // If the "Rightward Force" is too low (i.e. less than 100), the ball will
        // take an undesirable amount of time to travel across the canvas. In this
        // case, call reset() after 2 seconds. 
        if (abs(new_vx) < 100) {
            setTimeout(reset,2000);
        }
    }
}

 /**
 * This function changes the x and y values of the ball's velocity vector based on 
 * the acceleration vector, the damping factors (vMultiplier and bMultiplier), and 
 * whether the ball has passed the edges of the canvas.
 * Based on the code from: https://p5js.org/examples/simulate-smokeparticles.html
 */
function ballMove() {
	vx = vx + ax;
	vy = vy + ay;
	y = y + vy * vMultiplier; 
	x = x + vx * vMultiplier;
    // Change velocity vector when ball reaches edges of the canvas
    // Applies for the left edge of the canvas:
	if (x < 0) { 
        x = 0; 
        // Change direction of the x component of the velocity vector
		vx = -vx * bMultiplier; 
    }
    // Applies for the bottom of the canvas:
 	if (y < 0) { 
         y = 0; 
        // Change direction of the y component of the velocity vector
 		vy = -vy * bMultiplier; 
     }
    // Applies for the right edge of the canvas. If the ball reaches this edge, this
    // means it has reached the other side of the fire successfully. Call the 
    // list_append() and reset() functions to add the relevant data to "memory" and
    // train the neural network.
 	if (x > width - 20) { 
        survive = 1;
        list_append();
        reset();
     }
    // Applies for the top of the canvas:
 	if (y > height - 20) { 
        y = height - 20; 
        // Change direction of the y component of the velocity vector 
 		vy = -vy * bMultiplier; 
     }
}

//========= PARTICLE SYSTEM ===========

/**
 * A basic particle system class
 * @param num the number of particles
 * @param v the origin of the particle system
 * @param img_ a texture for each particle in the system
 * @constructor
 * Based on the code from: https://p5js.org/examples/simulate-smokeparticles.html
 */
let ParticleSystem = function(num,v,img_) {
    this.particles = [];
    // We make sure to copy the vector value in case we accidentally mutate the original by accident
    this.origin = v.copy(); 
    this.img = img_
    for(let i = 0; i < num; ++i){
        this.particles.push(new Particle(this.origin,this.img));
    }
};

/**
 * This function runs the entire particle system.
 */
ParticleSystem.prototype.run = function() {
    // cache length of the array we're going to loop into a variable
    let len = this.particles.length;

    // Loop through and run particles
    for (let i = len - 1; i >= 0; i--) {
        let particle = this.particles[i];
        particle.run();

        // If the particle is dead, we remove it using splice().
        if (particle.isDead()) {
            this.particles.splice(i,1);
        }
    }
}

/**
 * Method to add a force vector to all particles currently in the system
 * @param dir A p5.Vector describing the direction of the force.
 */
ParticleSystem.prototype.applyForce = function(dir) {
    let len = this.particles.length;
    for(let i = 0; i < len; ++i){
        this.particles[i].applyForce(dir);
    }
}

/**
 * Adds a new particle to the system at the origin of the system and with
 * the originally set texture.
 */
ParticleSystem.prototype.addParticle = function() {
    this.particles.push(new Particle(this.origin,this.img));
}

/**
 * If the ball contacts the fire, it did not successfully make it to the other side.
 * Call the list_append() and reset() functions to end the current session of the game,
 * add the relevant data to "memory," and train the neural network.
 */
ParticleSystem.prototype.contact = function() {
    let len = this.particles.length;
    let particles_y = [];
    let particles_x = [];
    // For every particle in the system, add the x and y values of the position vectors
    // to the "particles_y" and "particles_x" arrays. 
    for(let i = 0; i < len; ++i){
        particles_y.push(this.particles[i].loc.y)
        particles_x.push(this.particles[i].loc.x)
    }
    // Identify the minimum and maximum values of the "particles_y" and "particles_x"
    // arrays to identify the spatial extent of the fire.
    min_y = Math.min.apply(null,particles_y);
    min_x = Math.min.apply(null,particles_x);
    max_x = Math.max.apply(null,particles_x);
    // Test to see whether the ball's position vector locates it within the spatial
    // extent of the fire. If so, set "survive" to 0 and call list_append() and reset().
    if (y >= min_y && (x >= min_x && x <= max_x)) {
        survive = 0;
        list_append();
        reset();
    }
}

//========= PARTICLE  ===========

/**
 * A simple Particle class, renders the particle as an image
 * Based on the code from: https://p5js.org/examples/simulate-smokeparticles.html
 * @constructor
 */
let Particle = function (pos, img_) {
    this.loc = pos.copy();
    // Set the values of the velocity vector of the particle to random values. 
    let vx = randomGaussian() * 0.3;
    let vy = randomGaussian() * 0.3 - 1.0;

    this.vel = createVector(vx,vy);
    this.acc = createVector();
    this.lifespan = 100.0;
    this.texture = img_;
}

/**
 *  Simulataneously updates and displays a particle.
 */
Particle.prototype.run = function() {
    this.update();
    this.render();
}

/**
 *  A function to display a particle
 */
Particle.prototype.render = function() {
    imageMode(CENTER);
    tint(200,50,0,this.lifespan);
    image(this.texture,this.loc.x,this.loc.y);
}

/**
 * A method to apply a force vector to a particle.
 * @param f Force vector
 */
Particle.prototype.applyForce = function(f) {
    this.acc.add(f);
}

/**
 *  This method checks to see if the particle has reached the end of it's lifespan,
 *  if it has, return true, otherwise return false.
 */
Particle.prototype.isDead = function () {
    if (this.lifespan <= 0.0) {
        return true;
    } else {
        return false;
    }
}

/**
 *  This method updates the position of the particle.
 */
Particle.prototype.update = function() {
    this.vel.add(this.acc);
    this.loc.add(this.vel);
    this.lifespan -= 2.5;
    this.acc.mult(0);
}

function preload() {
    particle_texture = loadImage("particle_texture.png");
}

function setup() {
    // Set the canvas size
    createCanvas(page_width,page_height);
    // Initialize our particle system
    ps = new ParticleSystem(0,createVector(width / 2, height),particle_texture);
    // Set the initial values of the position vector of the ball so that it appears
    // in the bottom left corner of the canvas.
    x = 20;
    y = page_height;
    // Set the initial values of the velocity vector of the ball so that it appears
    // motionless at the start of the game.
    vx = 0;
    vy = 0;
    // Create the "Rightward Force" slider
    slider1 = createSlider(0,4000,1000,0);
    slider1.position(17,10);
    div1 = createDiv("Rightward Force");
    div1.position(35,25);
    // Create the "Upward Force" slider.
    div1.style("color:white");
    div2 = createDiv("Upward Force");
    div2.position(-5,90);
    div2.style("color:white");
    div2.style("rotate",270);
    // Create additional DOM elements
    div3 = createDiv("AI Learning...")
    div3.style("color:white");
    div3.position(550,20);
    div3.hide();
    whichDiv = createDiv("Player: You")
    whichDiv.position(width/2-20,20);
    whichDiv.style("color:white");
    slider2 = createSlider(-2000,0,-1000,0);
    slider2.position(-45,75);
    slider2.style("rotate",90);
    button1 = createButton("Let AI Try");
    button1.position(570,20);
    // Callback for "button1" once it is clicked
    // Once the "Let AI Try" button is pressed, the game switches the player to the AI.
    // Hide the "Let AI Try" button, change the title to "Player: AI" from "Player: You" 
    // and set "ai_ready" to 1 so that the "Go!" button will be displayed. 
    button1.mousePressed(function() {
        button_press = 1;
        button1.hide();
        whichDiv.remove();
        whichDiv = createDiv("Player: AI");
        whichDiv.position(width/2-20,20);
        whichDiv.style("color:white");
        ai_ready = 1;
    });
    button2 = createButton("Go!");
    button2.position(85,85);
    button2.mousePressed(go_button);
    // Explanatory text for the game
    explanationDiv1 = createDiv('Instructions for the Game:');
    explanationDiv2 = createDiv('The goal is to shoot the ball to the other')
    explanationDiv3 = createDiv('side. Hit the fire and you lose. Adjust')
    explanationDiv4 = createDiv('the "Upward Force" and "Rightward');
    explanationDiv5 = createDiv('Force" sliders to control the angle of');
    explanationDiv6 = createDiv('the shot. Press the "Go!" button to shoot.');
    explanationDiv7 = createDiv('There are two possible players: You and');
    explanationDiv8 = createDiv('an AI bot. Hit the "Let AI Try" button to');
    explanationDiv9 = createDiv('activate the AI. From now on when you');
    explanationDiv10 = createDiv('press the "Go!" button the AI will set');
    explanationDiv11 = createDiv('the sliders.')
    explanationDiv12 = createDiv('Watch over the first few tries as the AI');
    explanationDiv13 = createDiv('experiments and learns how to shoot the'); 
    explanationDiv14 = createDiv('ball to avoid the fire. It will quickly figure');
    explanationDiv15 = createDiv('out one way to shoot the ball successfully'); 
    explanationDiv16 = createDiv('no matter which way the fire is facing.');
    explanationDiv1.style('position',width+20,15); 
    explanationDiv2.style('position',width+20,35); 
    explanationDiv3.style('position',width+20,55); 
    explanationDiv4.style('position',width+20,75);
    explanationDiv5.style('position',width+20,95);
    explanationDiv6.style('position',width+20,115);
    explanationDiv7.style('position',width+20,145);
    explanationDiv8.style('position',width+20,165);
    explanationDiv9.style('position',width+20,185);
    explanationDiv10.style('position',width+20,205);
    explanationDiv11.style('position',width+20,225);
    explanationDiv12.style('position',width+20,255);
    explanationDiv13.style('position',width+20,275);
    explanationDiv14.style('position',width+20,295);
    explanationDiv15.style('position',width+20,315);
    explanationDiv16.style('position',width+20,335);
    // Set the values for the wind vector to random values 
    dx = random(-0.2,0.2);
    dy = random(0,-0.2);
    wind = createVector(dx,dy);
}

function draw() {
    background(0);
    // Show the "Go!" button if in "Player: You" mode
    if (button_press == 0) {
        button2.show();
    } else if (button_press == 1) {
        // Show the "Go!" button only if "ai_ready" is equal to 1
        if (ai_ready == 0) {
            button2.hide();
            div3.show();
        } else if (ai_ready == 1) {
            button2.show();
            div3.hide();
        }
    }
    // Apply wind force and run the particle system
    ps.applyForce(wind);
    ps.run();
    for (let i = 0; i < 2; i++) {
        ps.addParticle();
    }
    // Call the ballMove() and ps.contact() functions to move the ball. 
    // If the ball has reached the edges of the canvas or has touched the fire, end
    // the current session of the game.
    ballMove();
    ps.contact();
    // Draw the ball at the current x and y values of its position vector
    ellipse(x,y,30,30);
}

