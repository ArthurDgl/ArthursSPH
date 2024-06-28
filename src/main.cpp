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
    float kernelCutoffSquared;

    float EOSCoefficient;
    float EOSExponent;
    float EOSRestDensity;
};

void render(ProgramState* state);
void displayGui(ProgramState* state);

void initializeParticles(ProgramState* state);
void updateParticles(ProgramState* state);
void integrateParticles(ProgramState* state);
void applyBoundaryConditions(ProgramState* state);
void resetParticleForces(ProgramState* state);

void computeDensities(ProgramState* state);
float SPHDensity(sf::Vector2f position, Particle* particles, int particleAmount, float cutoff, float cutoffSquared);

void computePressures(ProgramState* state);
float EOSPressure(float density, float restDensity, float k, float gamma);

void applyPressureForces(ProgramState* state);
sf::Vector2f calculatePressureForce(int particleIndex, Particle *particles, int particleAmount, float cutoffSquared);

float SPHKernel(float distanceSquared, float cutoffSquared);
sf::Vector2f SPHKernelGradient(sf::Vector2f iPosition, sf::Vector2f jPosition, float cutoffSquared);

float randomFloat(float lowerBound, float upperBound); // From ChatGPT

int main() {
    ProgramState state = {
        .width = 1200,
        .height = 800,
        .particleAmount = 500,
        .kernelCutoff = 100.0f,
        .EOSCoefficient = 20.0f,
        .EOSExponent = 7.0f,
        .EOSRestDensity = 5.0f
    };
    state.kernelCutoffSquared = state.kernelCutoff * state.kernelCutoff;
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
    state->deltaTime = deltaTime.asSeconds();
    ImGui::SFML::Update(*state->window, state->deltaClock->restart());

    ImGui::Begin("SPH Configuration");
    float fps = std::round(100.0f/deltaTime.asSeconds()) / 100.0f;
    ImGui::Text((std::string("FPS : ") + std::to_string(fps)).c_str());

    ImGui::NewLine();
    ImGui::Text("Kernel");
    ImGui::InputFloat("Cutoff Distance", &state->kernelCutoff);
    state->kernelCutoffSquared = state->kernelCutoff * state->kernelCutoff;

    ImGui::NewLine();
    ImGui::Text("Equation of State");
    ImGui::InputFloat("Pressure Coefficient", &state->EOSCoefficient);
    ImGui::InputFloat("Density Exponent", &state->EOSExponent);
    ImGui::InputFloat("Rest Density", &state->EOSRestDensity);

    ImGui::End();
}


void initializeParticles(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i] = Particle{
            1.0f,
            sf::Vector2f(randomFloat(0, 1.0f*state->width), randomFloat(0, 1.0f*state->height)),
            sf::Vector2f(0, 0),
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

    integrateParticles(state);
    applyBoundaryConditions(state);
}


void integrateParticles(ProgramState* state) {
    float dT = state->deltaTime;
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].velocity += state->particles[i].forces * (dT / state->particles[i].mass);
        state->particles[i].position += state->particles[i].velocity * dT;
    }
}

void applyBoundaryConditions(ProgramState* state) {
    int r = state->particles->radius * 2;
    for (int i = 0; i < state->particleAmount; i++) {
        if (state->particles[i].position.x > state->width - r) state->particles[i].position.x = state->width - r;
        if (state->particles[i].position.y > state->height - r) state->particles[i].position.y = state->height - r;

        if (state->particles[i].position.x < 0) state->particles[i].position.x = 0.0f;
        if (state->particles[i].position.y < 0) state->particles[i].position.y = 0.0f;
    }
}


void resetParticleForces(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].forces *= 0.0f;
    }
}


void computeDensities(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].density = SPHDensity(state->particles[i].position, state->particles, state->particleAmount, state->kernelCutoff, state->kernelCutoffSquared);
    }
}


void computePressures(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].pressure = EOSPressure(state->particles[i].density, state->EOSRestDensity, state->EOSCoefficient, state->EOSExponent);
        state->particles[i].pressureOverDSquared = state->particles[i].pressure / (state->particles[i].density * state->particles[i].density);
    }
}


float EOSPressure(float density, float restDensity, float k, float gamma) {
    return k * (std::pow(density / restDensity, gamma) - 1);
}


float SPHDensity(sf::Vector2f position, Particle* particles, int particleAmount, float cutoff, float cutoffSquared) {
    float density = 0.0f;
    
    for (int j = 0; j < particleAmount; j++) {
        float dX = position.x - particles[j].position.x;
        float dY = position.y - particles[j].position.y;

        if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) continue;

        float distanceSquared = dX*dX + dY*dY;
        
        density += particles[j].mass * SPHKernel(distanceSquared, cutoffSquared);
    }

    return density;
}


void applyPressureForces(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].forces += calculatePressureForce(i, state->particles, state->particleAmount, state->kernelCutoffSquared);
    }
}


sf::Vector2f calculatePressureForce(int particleIndex, Particle *particles, int particleAmount, float cutoffSquared) {
    sf::Vector2f pressureForce = sf::Vector2f(0, 0);

    for (int j = 0; j < particleAmount; j++) {
        if (j == particleIndex) continue;

        float scalarPart = particles[j].mass * (particles[particleIndex].pressureOverDSquared + particles[j].pressureOverDSquared);
        pressureForce -= scalarPart * SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoffSquared);
    }

    return pressureForce;
}


float SPHKernel(float distanceSquared, float cutoffSquared) {
    if (distanceSquared >= cutoffSquared) return 0.0f;
    const float q = distanceSquared/cutoffSquared;
    return std::pow(q, 2.0f) - 2*q + 1.0f;
}


sf::Vector2f SPHKernelGradient(sf::Vector2f iPosition, sf::Vector2f jPosition, float cutoffSquared) {
    sf::Vector2f vectorPart = jPosition - iPosition;

    float distanceSquared = vectorPart.x * vectorPart.x + vectorPart.y * vectorPart.y;

    if (distanceSquared >= cutoffSquared) return sf::Vector2f(0, 0);

    float scalarPart = 4.0f / cutoffSquared * (distanceSquared / cutoffSquared - 1);

    return vectorPart * scalarPart;
}


float randomFloat(float lowerBound, float upperBound) { // From ChatGPT
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lowerBound, upperBound);
    return dis(gen);
}