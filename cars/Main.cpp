#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <array>


float dist(float x1, float y1, float x2, float y2) { return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)); }
sf::Vector2f addVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x + v2.x, v1.y + v2.y); }
sf::Vector2f subVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x - v2.x, v1.y - v2.y); }
sf::Vector2f multVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return sf::Vector2f(v1.x * v2.x, v1.y * v2.y); }
sf::Vector2f normalizeVector2f(sf::Vector2f v1) { float	d = dist(0, 0, v1.x, v1.y); return sf::Vector2f(v1.x / d, v1.y / d); }
float dotProductVectors2f(sf::Vector2f v1, sf::Vector2f v2) { return v1.x * v2.x + v1.y * v2.y; };
float angleVect(sf::Vector2f v) { if (v.x == 0) { return 3.141592 / 2 * abs(v.y) / v.y; }; if (v.y == 0) { return 3.141592 * abs(v.x) / v.x; }; return atan2(v.y, v.x); }
sf::Vector2f vectAngle(float a) { return sf::Vector2f(cos(a), sin(a)); }


void write(sf::RenderWindow& window, std::string t1, sf::IntRect rect, sf::Color col) {
	//load font
	sf::Font font;
	font.loadFromFile("../fonts/BalooPaaji2.ttf");

	//create the text element
	sf::Text text(t1, font);
	text.setStyle(sf::Text::Bold);
	text.setFillColor(col);

	text.setCharacterSize(rect.height);

	text.setPosition(sf::Vector2f(rect.left, rect.top));

	window.draw(text);
}


sf::Color HSVtoRGB(float H, float S, float V) {   // h:0-360.0, s:0.0-1.0, v:0.0-1.0
	S *= 100; V *= 100;
	if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
		std::cout << "The given HSV values are not in valid range" << "\n";
		return sf::Color(0, 0, 0);
	}
	float s = S / 100;
	float v = V / 100;
	float C = s * v;
	float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	float m = v - C;
	float r, g, b;
	if (H >= 0 && H < 60) {
		r = C, g = X, b = 0;
	}
	else if (H >= 60 && H < 120) {
		r = X, g = C, b = 0;
	}
	else if (H >= 120 && H < 180) {
		r = 0, g = C, b = X;
	}
	else if (H >= 180 && H < 240) {
		r = 0, g = X, b = C;
	}
	else if (H >= 240 && H < 300) {
		r = X, g = 0, b = C;
	}
	else {
		r = C, g = 0, b = X;
	}
	int R = (r + m) * 255;
	int G = (g + m) * 255;
	int B = (b + m) * 255;

	return sf::Color(R, G, B);
}



class Layer {
public:
	int size;
	int preSize;
	std::vector<float> values;
	std::vector<std::vector<float>> weights;
	std::vector<float> biases;

	void init(int s, int ps);
	void randomize();
	//void copyLayerAtPrcnt(Layer L, int prcnt);
};

void Layer::init(int s, int ps) {
	size = s; preSize = ps;
	values.clear();
	biases.clear();
	std::vector<float> v;
	for (int a = 0; a < ps; a++) {
		v.push_back(0);
	}

	for (int i = 0; i < s; i++) {
		values.push_back(0);
		biases.push_back(0);
		weights.push_back(v);
	}
}

void Layer::randomize() {
	for (int i = 0; i < size; i++) {
		//values[i] = (rand()%1000-500)/100;
		biases[i] = (rand()%1000-500)/100;
		for (int a = 0; a < preSize; a++) {
			weights[i][a] = (rand()%1000-500)/200;
		}
	}
}


struct NeuralNet {
	std::vector<Layer> layers;
};


void updateLayer(Layer pre, Layer& lay) {
	for (int i = 0; i < lay.size; i++) {
		lay.values[i] = lay.biases[i];
		for (int j = 0; j < pre.size; j++) {
			lay.values[i] += pre.values[j] * lay.weights[i][j];
		}

		// faire activation
		// tanh
		lay.values[i] = std::tanh(lay.values[i]);
	}

}


Layer runNN(NeuralNet NN) {
	for (int l = 1; l < NN.layers.size(); l++) {
		updateLayer(NN.layers[l - 1], NN.layers[l]);
	}

	return NN.layers[NN.layers.size() - 1];
}




class Checkpoint {
public:
	sf::RectangleShape rect;
	sf::Color color = sf::Color::Red;
	int numero;

	std::array<sf::Vector2f, 4> pointsPosition;

	void init(sf::Vector2f center, sf::Vector2f size, float orientation, int num);
};

void Checkpoint::init(sf::Vector2f center, sf::Vector2f size, float orientation, int num) {
	//create rect
	rect.setSize(size);
	rect.setOrigin(sf::Vector2f(size.x / 2, size.y / 2));
	rect.setRotation(orientation * 57.29);
	rect.setPosition(center);
	rect.setFillColor(color);

	// checkpoint's index
	numero = num;

	//fill points position
	sf::Vector2f v1 = vectAngle(orientation);
	sf::Vector2f v2 = sf::Vector2f(v1.y * -1, v1.x); //v1 + pi/2

	//scale the vectors so they can reach the angles of the rect
	v1.x *= size.x / 2; v1.y *= size.x / 2;
	v2.x *= size.y / 2; v2.y *= size.y / 2;

	sf::Vector2f p1, p2;
	pointsPosition[0] = addVectors2f(addVectors2f(center, v1), v2);
	pointsPosition[1] = subVectors2f(addVectors2f(center, v1), v2);
	pointsPosition[2] = subVectors2f(subVectors2f(center, v1), v2);
	pointsPosition[3] = addVectors2f(subVectors2f(center, v1), v2);
}


bool pointInRect(sf::Vector2f& p1, Checkpoint& c) {
	//check if p1 is in a checkpoint
	sf::Vector2f a, b, d; a = c.pointsPosition[0]; b = c.pointsPosition[1]; d = c.pointsPosition[3];
	float bax = b.x - a.x;
	float bay = b.y - a.y;
	float dax = d.x - a.x;
	float day = d.y - a.y;

	if ((p1.x - a.x) * bax + (p1.y - a.y) * bay < 0) return false;
	if ((p1.x - b.x) * bax + (p1.y - b.y) * bay > 0) return false;
	if ((p1.x - a.x) * dax + (p1.y - a.y) * day < 0) return false;
	if ((p1.x - d.x) * dax + (p1.y - d.y) * day > 0) return false;

	return true;
}



class Car {
public:
	sf::Vector2f position;
	sf::Vector2f speedVect;
	float maxspeed;
	sf::Vector2f orientation;

	sf::Vector2f size;
	sf::RectangleShape rect;

	int lastCheckpoint;

	NeuralNet NN;
	bool isdead = false;
	float score;
	float life; // so the cars don't stay alive for too long, life is generated by going though a checkpoint
	const float maxlife = 100;
	const float viewDist = 300;

	void init(sf::Vector2f pos, sf::Vector2f ori);
	void update(sf::Image& Img, std::vector<Checkpoint>& checkpoints);
	void draw(sf::RenderWindow& window);
	float distWall(float angle, sf::Image& Img); //angle = difference with orientation
};


void Car::init(sf::Vector2f pos, sf::Vector2f ori) {
	isdead = false;
	score = 0;
	lastCheckpoint = 0;

	position = pos;
	orientation = ori;
	speedVect = sf::Vector2f(0, 0);
	maxspeed = 1;
	life = maxlife;

	rect.setFillColor(HSVtoRGB(rand()% 360, 1, 1));
	size = sf::Vector2f(30, 10);
	rect.setSize(size);
	rect.setOrigin(sf::Vector2f(15, 5));
	rect.setRotation(angleVect(orientation) * 57.29);
	rect.setRotation(3.141592 / 2);

	Layer l1, l2, l3, l4, l5;
	l1.init(10, 0);
	l2.init(5, 10);
	l3.init(5, 5);
	l4.init(3, 5);
	l5.init(2, 3);
	l1.randomize();
	l2.randomize();
	l3.randomize();
	l4.randomize();
	l5.randomize();
	NN.layers.push_back(l1);
	NN.layers.push_back(l2);
	NN.layers.push_back(l3);
	NN.layers.push_back(l4);
	NN.layers.push_back(l5);
}


float Car::distWall(float angle, sf::Image& Img) {
	float a = angleVect(orientation) + angle;

	sf::Vector2f tp;
	sf::Vector2f vect = sf::Vector2f(cos(a), sin(a));
	float dist = 5;
	bool test = false;

	while (dist < viewDist and !test) {
		dist *= 1.5;
		tp.x = vect.x * dist + position.x;
		tp.y = vect.y * dist + position.y;
		test = Img.getPixel(fmax(0, fmin(tp.x, 1919)), fmax(0, fmin(tp.y, 1079))) == sf::Color(0, 0, 0);
	}



	return dist;
}


void Car::update(sf::Image& Img, std::vector<Checkpoint>& checkpoints) {
	isdead = isdead or (life <= 0);

	if (!isdead) {
		NN.layers[0].values[0] = distWall(-3.141592 / 3, Img);
		NN.layers[0].values[1] = distWall(-3.141592 / 4, Img);
		NN.layers[0].values[2] = distWall(-3.141592 / 6, Img);
		NN.layers[0].values[3] = distWall(0, Img);
		NN.layers[0].values[4] = distWall(3.141592 / 6, Img);
		NN.layers[0].values[5] = distWall(3.141592 / 4, Img);
		NN.layers[0].values[6] = distWall(3.141592 / 3, Img);
		NN.layers[0].values[7] = angleVect(orientation) / 10;
		NN.layers[0].values[8] = speedVect.x / 5;
		NN.layers[0].values[9] = speedVect.y / 5;

		Layer ans = runNN(NN);

		//add 0.1 to speed so the cars can't stay at the same place
		float speedC = (0.1 + abs(ans.values[0])/2);

		float angle = ans.values[1] / 10; //turn angle
		orientation = vectAngle(angleVect(orientation) + angle);

		speedVect.x += speedC * orientation.x * maxspeed;
		speedVect.y += speedC * orientation.y * maxspeed;

		speedVect.x *= 0.9; speedVect.y *= 0.9;

		position = addVectors2f(position, speedVect);

		//test collision
		sf::Vector2f v1 = orientation;
		sf::Vector2f v2 = sf::Vector2f(orientation.y * -1, orientation.x); //v1 + pi/2

		//scale the vectors so they can reach the angles of the rect
		v1.x *= size.x / 2; v1.y *= size.x / 2;
		v2.x *= size.y / 2; v2.y *= size.y / 2;

		sf::Vector2f p1, p2;
		sf::Color col;
		// both front angles
		p1 = addVectors2f(addVectors2f(position, v1), v2);
		col = Img.getPixel(fmax(0, fmin(p1.x, 1920)), fmax(0, fmin(p1.y, 1080)));
		if (col == sf::Color(0, 0, 0)) { isdead = true; score += (maxlife - life)/2; }

		p2 = subVectors2f(addVectors2f(position, v1), v2);
		col = Img.getPixel(fmax(0, fmin(p2.x, 1920)), fmax(0, fmin(p2.y, 1080)));
		if (col == sf::Color(0, 0, 0)) { isdead = true; score += (maxlife - life)/2; }

		/* // back angles, if the car can go backward
		p = addVectors2f(subVectors2f(position, v1), v2);
		col = Img.getPixel(fmax(0, fmin(p.x, 1920)), fmax(0, fmin(p.y, 1080)));
		if (col == sf::Color(0, 0, 0)) { isdead = true; score += (maxlife - life)/2; }
		p = subVectors2f(subVectors2f(position, v1), v2);
		col = Img.getPixel(fmax(0, fmin(p.x, 1920)), fmax(0, fmin(p.y, 1080)));
		if (col == sf::Color(0, 0, 0)) { isdead = true; score += (maxlife - life)/2; }
		*/

		if (lastCheckpoint < checkpoints.size()) {
			if (pointInRect(p1, checkpoints[lastCheckpoint]) or pointInRect(p2, checkpoints[lastCheckpoint])) {
				score += life;
				life = maxlife;
				lastCheckpoint += 1;
				if (lastCheckpoint == checkpoints.size()) {
					isdead = true;
				}
			}
		}

		life -= !isdead * maxspeed / 3;
	}
}



void Car::draw(sf::RenderWindow& window) {
	rect.setRotation(angleVect(orientation) * 57.29);
	rect.setPosition(position);

	window.draw(rect);
}



void saveBestCar(std::vector<Car>& cars, Car& BestCar) {
	float maxScore = 0; float indexMax = 0;

	for (int i = 0; i < cars.size(); i++) {
		if (cars[i].score > maxScore) {
			indexMax = i;
			maxScore = cars[i].score;
		}
	}

	BestCar.NN = cars[indexMax].NN;
	BestCar.score = cars[indexMax].score;
}


void evolveNN(NeuralNet& NN, float percent) {
	for (int layer = 0; layer < NN.layers.size(); layer++) {

		for (int e = 0; e < NN.layers[layer].size; e++) {
			if (percent > rand() % 1000 / 10) {
				NN.layers[layer].biases[e] = (rand()%1000 - 500)/200;
			}
			for (int w = 0; w < NN.layers[layer].preSize; w++) {
				if (percent > rand() % 1000 / 10) {
					NN.layers[layer].weights[e][w] = (rand()%1000 - 500)/200;
				}
			}
		}
	}
}


void updateCars(std::vector<Car>& cars, sf::Image mapImg, std::vector<Checkpoint>& checkpoints, Car& BestCar, bool& end) {
	bool alldead = true;
	for (int i = 0; i < cars.size(); i++) {
		cars[i].update(mapImg, checkpoints);
		alldead = alldead & cars[i].isdead;
	}

	if (alldead) {
		saveBestCar(cars, BestCar);
		end = true;
	}
}

void drawCars(std::vector<Car> cars, sf::RenderWindow& window) {
	for (int i = 0; i < cars.size(); i++) {
		if (!cars[i].isdead) {
			cars[i].draw(window);
		}
	}
}

void drawCheckpoints(std::vector<Checkpoint>& checkpoints, sf::RenderWindow& window) {
	for (int i = 0; i < checkpoints.size(); i++) {
		window.draw(checkpoints[i].rect);
	}
}

void initCheckpoints(std::vector<Checkpoint>& checkpoints, int num) {
	checkpoints.clear();
	Checkpoint ch;
	if (num == 1) {
		ch.init(sf::Vector2f(400, 230), sf::Vector2f(200, 15), 3.141592 / 4, 1);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(980, 260), sf::Vector2f(210, 15), 3.141592 / 1.9, 2);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(1550, 375), sf::Vector2f(300, 15), 3.141592 / -4, 3);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(1521, 800), sf::Vector2f(300, 15), 0.1, 4);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(990, 890), sf::Vector2f(200, 15), 3.141592 / 2, 5);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(440, 870), sf::Vector2f(200, 15), -3.141592 / 4, 6);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(405, 575), sf::Vector2f(120, 40), -0.1, 7);
		checkpoints.push_back(ch);
	}
	else {
		//map2.png
		ch.init(sf::Vector2f(400, 230), sf::Vector2f(200, 15), 3.141592 / 4, 1);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(1083, 189), sf::Vector2f(210, 15), 3.141592 / 1.9, 2);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(1600, 375), sf::Vector2f(470, 15), 3.141592 / -4, 3);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(1571, 910), sf::Vector2f(135, 15), 0.1, 4);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(980, 800), sf::Vector2f(120, 15), 3.141592 / 2, 5);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(400, 875), sf::Vector2f(170, 15), -3.141592 / 4, 6);
		checkpoints.push_back(ch);
		ch.init(sf::Vector2f(410, 555), sf::Vector2f(120, 30), -0.15, 7);
		checkpoints.push_back(ch);
	}
}




int main() {
	const float EVOLUTION = 5; //avg prcnt change of each generation
	const bool CHANGEMAP = true;

	std::srand(std::time(nullptr));

	int windowWidth = 1920;
	int windowHeight = 1080;

	sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Name", sf::Style::Fullscreen);
	sf::Texture mapTxt;
	if (!mapTxt.loadFromFile("../img/map1Texture.png")) return -1;
	sf::Image mapImg;
	if (!mapImg.loadFromFile("../img/map1.png")) return -1;
	sf::RectangleShape Map;
	Map.setTexture(&mapTxt);
	Map.setSize(sf::Vector2f(1920, 1080));
	Map.setPosition(sf::Vector2f(0, 0));
	window.draw(Map);


	std::vector<Car> cars;
	Car BestCar;
	BestCar.init(sf::Vector2f(400, 500), sf::Vector2f(0, -1));

	for (int i = 0; i < 5000; i++) {
		Car c;
		c.init(sf::Vector2f(400, 500), sf::Vector2f(0, -1));
		cars.push_back(c);
	}

	std::vector<Checkpoint> checkpoints;
	initCheckpoints(checkpoints, 1);


	//texts rect
	sf::IntRect genRect;
	genRect.top = 7; genRect.left = 20; genRect.height = 40; genRect.width = 240;
	sf::IntRect scoreRect = genRect;
	scoreRect.top += scoreRect.height * 1;

	int generation = 0;
	std::cout << "Generation n�: " << generation << '\n';
	bool endOfGeneration = false;

	while (window.isOpen()) {

		if (endOfGeneration) {
			for (int i = 0; i < cars.size(); i++) {
				cars[i].init(BestCar.position, BestCar.orientation);
				cars[i].NN = BestCar.NN;
				evolveNN(cars[i].NN, EVOLUTION);
			}
			generation++;
			std::cout << "Generation n�: " << generation << "Best score: "<< BestCar.score << '\n';
			endOfGeneration = false;

			if (CHANGEMAP) {
				std::string adress = "../img/map";
				adress += std::to_string(generation % 2 + 1);

				if (!mapTxt.loadFromFile(adress+"Texture.png")) return -1;
				if (!mapImg.loadFromFile(adress+".png")) return -1;
				initCheckpoints(checkpoints, generation % 2 + 1);

				Map.setTexture(&mapTxt);
			}
		}


		std::thread tup(updateCars, std::ref(cars), mapImg, std::ref(checkpoints), std::ref(BestCar), std::ref(endOfGeneration));
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
		}

		window.clear();
		window.draw(Map);
		drawCars(cars, window);
		//drawCheckpoints(checkpoints, window);
		write(window, "Generation n�"+std::to_string(generation), genRect, sf::Color::Black);
		if(BestCar.score > 0)	write(window, "Previous best: " + std::to_string((int)(BestCar.score*3.14159)), scoreRect, sf::Color::Black); //don't display the true score because it's not precise enough when rounded

		tup.join();
		window.display();

	}
	return 0;
};