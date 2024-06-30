#include <iostream>
#include <SFML/Graphics.hpp>
#include <../../../../libs/imgui/imgui.h>
#include <../../../../libs/imgui-sfml/imgui-SFML.h>
#include <cmath>
#include <random>

struct Particle {
    float mass;
    sf::Vector2f position;
    sf::Vector2f velocity;

    sf::Vector2f forces;

    int radius;

    float density;
    float pressure;
    float pressureOverDSquared;
};

struct ProgramState {
    unsigned int width;
    unsigned int height;

    sf::RenderWindow* window;

    sf::Clock* deltaClock;
    float deltaTime;

    int particleAmount;
    Particle* particles;

    float kernelCutoff;

    float EOSCoefficient;
    float EOSExponent;
    float EOSRestDensity;

    float viscosityCoefficient;

    float gravitationalPull;

    float wallPushCutoff;
    float wallPushStrength;

    float inflowSpeed;
};

void render(ProgramState* state);
void displayGui(ProgramState* state);

void initializeParticles(ProgramState* state);
void updateParticles(ProgramState* state);
void integrateParticles(ProgramState* state);
void resetParticleForces(ProgramState* state);

void applyBoundaryConditions(ProgramState* state);
void applyWallForces(ProgramState* state);

void computeDensities(ProgramState* state);
float SPHDensity(sf::Vector2f position, Particle* particles, int particleAmount, float cutoff);

void computePressures(ProgramState* state);
float EOSPressure(float density, float restDensity, float k, float gamma);

void applyPressureForces(ProgramState* state);
void calculatePressureForce(int particleIndex, Particle* particles, int particleAmount, float cutoff);

void applyViscosityForces(ProgramState* state);
void calculateViscosityForce(int particleIndex, Particle* particles, int particleAmount, float cutoff, float viscosityCoefficient);

float SPHKernel(float distance, float cutoff);
sf::Vector2f SPHKernelGradient(sf::Vector2f iPosition, sf::Vector2f jPosition, float cutoff);
float SPHKernelLaplacian(float distance, float cutoff);

float randomFloat(float lowerBound, float upperBound); // From ChatGPT

int main() {
    ProgramState state = {
        .width = 800,
        .height = 300,
        .particleAmount = 400,
        .kernelCutoff = 100.0f,
        .EOSCoefficient = 100000.0f,
        .EOSExponent = 1.0f,
        .EOSRestDensity = 3.0f,
        .viscosityCoefficient = 1000.0f,
        .gravitationalPull = 0.0f,//100.0f,
        .wallPushCutoff = 30.0f,
        .wallPushStrength = 300.0f,
        .inflowSpeed = 100.0f
    };
    state.particles = new Particle[state.particleAmount];

    initializeParticles(&state);

    state.window = new sf::RenderWindow{ { state.width, state.height }, "ArthursSPH" };
    state.window->setFramerateLimit(144);

    state.deltaClock = new sf::Clock();

    std::cout << "ImGui Init : " << ImGui::SFML::Init(*state.window) << "\n";

    while (state.window->isOpen()) {
        for (auto event = sf::Event{}; state.window->pollEvent(event);) {
            ImGui::SFML::ProcessEvent(*state.window, event);
            if (event.type == sf::Event::Closed) {
                state.window->close();
            }
        }

        updateParticles(&state);
        render(&state);

        std::cout << state.particles[0].density << "\n";
    }

    ImGui::SFML::Shutdown();
    return 0;
}


void render(ProgramState *state) {
    displayGui(state);
    state->window->clear();

    sf::CircleShape circle = sf::CircleShape(state->particles->radius);
    sf::Color color = sf::Color::White;

    // Render Code here
    for (int i = 0; i < state->particleAmount; i++) {
        circle.setPosition(state->particles[i].position);
        if (state->particles[i].density < state->EOSRestDensity) color = sf::Color::Green;
        else color = sf::Color::Red;
        circle.setFillColor(color);
        state->window->draw(circle);
    }

    ImGui::SFML::Render(*state->window);
    state->window->display();
}


void displayGui(ProgramState *state) {
    sf::Time deltaTime = state->deltaClock->getElapsedTime();
    state->deltaTime = 0.01f;//deltaTime.asSeconds();
    ImGui::SFML::Update(*state->window, state->deltaClock->restart());

    ImGui::Begin("SPH Configuration");
    float fps = std::round(100.0f/deltaTime.asSeconds()) / 100.0f;
    ImGui::Text((std::string("FPS : ") + std::to_string(fps)).c_str());

    ImGui::NewLine();
    ImGui::Text("Kernel");
    ImGui::InputFloat("Cutoff Distance", &state->kernelCutoff);

    ImGui::NewLine();
    ImGui::Text("Equation of State");
    ImGui::InputFloat("Pressure Coefficient", &state->EOSCoefficient);
    ImGui::InputFloat("Density Exponent", &state->EOSExponent);
    ImGui::InputFloat("Rest Density", &state->EOSRestDensity);

    ImGui::NewLine();
    ImGui::Text("Viscosity");
    ImGui::InputFloat("Viscosity Coefficient", &state->viscosityCoefficient);

    ImGui::NewLine();
    ImGui::Text("Gravity");
    ImGui::InputFloat("Gravitational Pull", &state->gravitationalPull);

    ImGui::End();
}


void initializeParticles(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i] = Particle{
            1.0f,
            sf::Vector2f(randomFloat(0, 1.0f*state->width), randomFloat(0, 1.0f*state->height)),
            sf::Vector2f(state->inflowSpeed, 0),
            sf::Vector2f(0, 0),
            5
        };
    }
}


void updateParticles(ProgramState* state) {
    resetParticleForces(state);

    computeDensities(state);
    computePressures(state);

    applyPressureForces(state);
    applyViscosityForces(state);

    applyWallForces(state);

    integrateParticles(state);
    applyBoundaryConditions(state);
}


void integrateParticles(ProgramState* state) {
    float dT = state->deltaTime;
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].velocity += state->particles[i].forces * (dT / state->particles[i].density);
        state->particles[i].position += state->particles[i].velocity * dT;
    }
}

void applyBoundaryConditions(ProgramState* state) {
    int r = state->particles->radius * 2;
    for (int i = 0; i < state->particleAmount; i++) {
        if (state->particles[i].position.x > state->width) {
            state->particles[i].position.x = -2*state->particles[i].radius;
        }
        if (state->particles[i].position.y > state->height - r) state->particles[i].position.y = state->height - r;

        if (state->particles[i].position.x < 0) state->particles[i].velocity = {state->inflowSpeed, 0.0f};
        if (state->particles[i].position.y < 0) state->particles[i].position.y = 0.0f;
    }
}

void applyWallForces(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        sf::Vector2f wallForce = sf::Vector2f(0, 0);
        if (state->particles[i].position.x < 0 + state->wallPushCutoff) wallForce.x += state->wallPushStrength;
        if (state->particles[i].position.y < 0 + state->wallPushCutoff) wallForce.y += state->wallPushStrength;
        //if (state->particles[i].position.x > state->width - state->wallPushCutoff) wallForce.x -= state->wallPushStrength;
        if (state->particles[i].position.y > state->height - state->wallPushCutoff) wallForce.y -= state->wallPushStrength;

        state->particles[i].forces += wallForce;
    }
}


void resetParticleForces(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].forces = sf::Vector2f(0.0f, state->gravitationalPull);
    }
}


void computeDensities(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].density = SPHDensity(state->particles[i].position, state->particles, state->particleAmount, state->kernelCutoff);
    }
}


void computePressures(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].pressure = EOSPressure(state->particles[i].density, state->EOSRestDensity, state->EOSCoefficient, state->EOSExponent);
        state->particles[i].pressureOverDSquared = state->particles[i].pressure / (state->particles[i].density * state->particles[i].density);
    }
}


float EOSPressure(float density, float restDensity, float k, float gamma) {
    return -k * (std::pow(density / restDensity, gamma) - 1);
}


float SPHDensity(sf::Vector2f position, Particle* particles, int particleAmount, float cutoff) {
    float density = 0.0f;
    
    for (int j = 0; j < particleAmount; j++) {
        float dX = position.x - particles[j].position.x;
        float dY = position.y - particles[j].position.y;

        if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) continue;

        float distance = std::sqrt(dX*dX + dY*dY);
        
        density += particles[j].mass * SPHKernel(distance, cutoff);
    }

    return density;
}


void applyPressureForces(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        calculatePressureForce(i, state->particles, state->particleAmount, state->kernelCutoff);
    }
}


void calculatePressureForce(int particleIndex, Particle* particles, int particleAmount, float cutoff) {
    for (int j = 0; j < particleAmount; j++) {
        if (j == particleIndex) continue;

        float scalarPart = particles[j].mass * (particles[particleIndex].pressureOverDSquared + particles[j].pressureOverDSquared);
        sf::Vector2f pressureForce = scalarPart * SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff);
        particles[particleIndex].forces -= pressureForce;
        particles[j].forces += pressureForce;
    }
}


void applyViscosityForces(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        calculateViscosityForce(i, state->particles, state->particleAmount, state->kernelCutoff, state->viscosityCoefficient);
    }
}



void calculateViscosityForce(int particleIndex, Particle* particles, int particleAmount, float cutoff, float viscosityCoefficient) {
    for (int j = 0; j < particleAmount; j++) {
        if (j == particleIndex) continue;

        float dX = particles[particleIndex].position.x - particles[j].position.x;
        float dY = particles[particleIndex].position.y - particles[j].position.y;

        if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) continue;

        float distance = std::sqrt(dX*dX + dY*dY);
        if (distance == 0.0f) continue;

        sf::Vector2f viscosityForce = viscosityCoefficient * particles[j].mass * SPHKernelLaplacian(distance, cutoff) / particles[j].density * (particles[j].velocity - particles[particleIndex].velocity);

        particles[particleIndex].forces += viscosityForce;
        particles[j].forces -= viscosityForce;
    }
}



float SPHKernel(float distance, float cutoff) {
    if (distance >= cutoff) return 0.0f;
    const float q = distance/cutoff;
    return std::pow(1 - q, 5.0f);
}


sf::Vector2f SPHKernelGradient(sf::Vector2f iPosition, sf::Vector2f jPosition, float cutoff) {
    sf::Vector2f vectorPart = jPosition - iPosition;

    if (std::abs(vectorPart.x) >= cutoff || std::abs(vectorPart.y) >= cutoff) return {0.0f, 0.0f};

    float distance = std::sqrt(vectorPart.x * vectorPart.x + vectorPart.y * vectorPart.y);

    if (distance >= cutoff || distance == 0.0f) return {0, 0};

    float scalarPart = -5 / (distance * cutoff) * std::pow(1 - distance / cutoff, 4.0f);

    return vectorPart * scalarPart;
}


float SPHKernelLaplacian(float distance, float cutoff) {
    if (distance >= cutoff) return 0.0f;
    float inverseCutoff = 1 / cutoff;
    return 20 * inverseCutoff * inverseCutoff * std::pow(1 - distance * inverseCutoff, 3.0f);
}


float randomFloat(float lowerBound, float upperBound) { // From ChatGPT
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lowerBound, upperBound);
    return dis(gen);
}