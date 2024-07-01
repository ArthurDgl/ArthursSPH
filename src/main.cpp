#include <iostream>
#include <SFML/Graphics.hpp>
#include <../../../../libs/imgui/imgui.h>
#include <../../../../libs/imgui-sfml/imgui-SFML.h>
#include <cmath>
#include <random>
#include <omp.h>

#include "../cmake-build-debug/_deps/sfml-src/extlibs/headers/glad/include/glad/gl.h"

struct Particle {
    float mass;
    sf::Vector2f position;
    sf::Vector2f velocity;

    sf::Vector2f forces;

    int radius;

    float density;
    float pressure;
    float pressureOverDSquared;

    int partitionIndex;
};

struct SphereObstacle {
    sf::Vector2f position;
    float radius;
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

    std::vector<int>* partition;
    int partitionSize;
    int partitionResolutionX;
    int partitionResolutionY;

    SphereObstacle obstacle;
};

void render(ProgramState* state);
void displayGui(ProgramState* state);

void initializePartition(ProgramState* state);
void clearPartition(ProgramState* state);
int getPartitionIndex(sf::Vector2f position, ProgramState* state);
void partitionParticles(ProgramState* state);

void initializeParticles(ProgramState* state);
void updateParticles(ProgramState* state);
void integrateParticles(ProgramState* state);
void resetParticleForces(ProgramState* state);

void applyBoundaryConditions(ProgramState* state);
void applyWallForces(ProgramState* state);

void computeDensities(ProgramState* state);
float SPHDensity(sf::Vector2f position, ProgramState* state);
float SPHDensitySingle(sf::Vector2f position, Particle* particles, int j, float cutoff);

void computePressures(ProgramState* state);
float EOSPressure(float density, float restDensity, float k, float gamma);

void applyPressureForces(ProgramState* state);
void calculatePressureForce(int particleIndex, ProgramState* state);
void calculatePressureForceSingle(int particleIndex, int j, Particle* particles, float cutoff);

void applyViscosityForces(ProgramState* state);
void calculateViscosityForce(int particleIndex, ProgramState* state);
void calculateViscosityForceSingle(int particleIndex, int j, Particle* particles, float cutoff, float viscosityCoefficient);

float SPHKernel(float distance, float cutoff);
sf::Vector2f SPHKernelGradient(sf::Vector2f iPosition, sf::Vector2f jPosition, float cutoff);
float SPHKernelLaplacian(float distance, float cutoff);

float randomFloat(float lowerBound, float upperBound); // From ChatGPT
float dotProduct(sf::Vector2f v1, sf::Vector2f v2);

int main() {
    ProgramState state = {
        .width = 800,
        .height = 300,
        .particleAmount = 10000,
        .kernelCutoff = 20.0f,
        .EOSCoefficient = 200000.0f,
        .EOSExponent = 1.0f,
        .EOSRestDensity = 0.5f,
        .viscosityCoefficient = 1000.0f,
        .gravitationalPull = 0.0f,//100.0f,
        .wallPushCutoff = 20.0f,
        .wallPushStrength = 8000.0f,
        .inflowSpeed = 500.0f,
        .obstacle = SphereObstacle{sf::Vector2f(200, 150), 50}
    };
    state.particles = new Particle[state.particleAmount];

    initializeParticles(&state);

    initializePartition(&state);

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
    }

    ImGui::SFML::Shutdown();
    return 0;
}


void render(ProgramState *state) {
    displayGui(state);
    state->window->clear();

    sf::CircleShape circle = sf::CircleShape(state->particles->radius);
    sf::Color color = sf::Color::Black;

    // Render Code here
    for (int i = 0; i < state->particleAmount; i++) {
        circle.setPosition(state->particles[i].position);
        float t = (state->particles[i].velocity.x / (state->inflowSpeed * 1.2f) + 1.0f) / 2.0f;
        if (t < 0) t = 0.0f;
        if (t > 1) t = 1.0f;
        color.r = 255 * t;
        color.g = 255 * (1 - t);
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


void initializePartition(ProgramState* state) {
    state->partitionResolutionX = state->width / state->kernelCutoff + 2;
    state->partitionResolutionY = state->height / state->kernelCutoff + 2;

    state->partitionSize = state->partitionResolutionX * state->partitionResolutionY;
    state->partition = new std::vector<int>[state->partitionSize];

    for (int i = 0; i < state->partitionSize; i++) {
        state->partition[i] = std::vector<int>();
        state->partition[i].clear();
    }
}

void clearPartition(ProgramState* state) {
    for (int i = 0; i < state->partitionSize; i++) {
        state->partition[i].clear();
    }
}

int getPartitionIndex(sf::Vector2f position, ProgramState* state) {
    int xIndex = position.x / state->kernelCutoff + 1;
    int yIndex = position.y / state->kernelCutoff + 1;

    return xIndex * state->partitionResolutionY + yIndex;
}

void partitionParticles(ProgramState *state) {
    for (int i = 0; i < state->particleAmount; i++) {
        int index = getPartitionIndex(state->particles[i].position, state);
        if (index < 0 ||index >= state->partitionSize) {
            std::cout << state->particles[i].position.x << "," << state->particles[i].position.y << "\n";
            index = 0;
        }

        state->particles[i].partitionIndex = index;
        state->partition[index].push_back(i);
    }
}


void initializeParticles(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i] = Particle{
            1.0f,
            sf::Vector2f(randomFloat(0, 1.0f*state->width), randomFloat(0, 1.0f*state->height)),
            sf::Vector2f(state->inflowSpeed, 0),
            sf::Vector2f(0, 0),
            2
        };
    }
}


void updateParticles(ProgramState* state) {
    clearPartition(state);
    partitionParticles(state);

    resetParticleForces(state);

    computeDensities(state);
    computePressures(state);

    applyPressureForces(state);
    applyViscosityForces(state);

    applyWallForces(state);

    applyBoundaryConditions(state);
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
    for (int i = 0; i < state->particleAmount; i++) {
        if (state->particles[i].position.x > state->width) {
            state->particles[i].position.x -= state->width + state->kernelCutoff/2.0f;
            //state->particles[i].position.y = randomFloat(state->wallPushCutoff, state->height - state->wallPushCutoff);
        }
        if (state->particles[i].position.y > state->height) state->particles[i].position.y = state->height;
        if (state->particles[i].position.x < -state->kernelCutoff) state->particles[i].position.x = -state->kernelCutoff;

        if (state->particles[i].position.x < 20 || state->particles[i].position.x > state->width - 20) state->particles[i].velocity = {state->inflowSpeed, 0.0f};
        if (state->particles[i].position.y < 0) state->particles[i].position.y = 0.0f;

        float sphereDist = sqrt(pow(state->particles[i].position.x - state->obstacle.position.x, 2.0f)
            + pow(state->particles[i].position.y - state->obstacle.position.y, 2.0f));
        float error = state->particles[i].radius + state->obstacle.radius - sphereDist;
        if (error > 0) {
            state->particles[i].position += (state->particles[i].position - state->obstacle.position) * (error / sphereDist);
        }
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
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].density = SPHDensity(state->particles[i].position, state);
    }
}


void computePressures(ProgramState *state) {
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].pressure = EOSPressure(state->particles[i].density, state->EOSRestDensity, state->EOSCoefficient, state->EOSExponent);
        state->particles[i].pressureOverDSquared = state->particles[i].pressure / (state->particles[i].density * state->particles[i].density);
    }
}


float EOSPressure(float density, float restDensity, float k, float gamma) {
    return -k * (std::pow(density / restDensity, gamma) - 1);
    //return -k * (density - restDensity);
}


float SPHDensity(sf::Vector2f position, ProgramState* state) {
    float density = 0.0f;

    int center = getPartitionIndex(position, state);

    for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
            int vectorIndex = center + k*state->partitionResolutionY + l;
            if (vectorIndex < 0 || vectorIndex >= state->partitionSize) continue;

            for (int m = 0; m < state->partition[vectorIndex].size(); m++) {
                density += SPHDensitySingle(position, state->particles, state->partition[vectorIndex].at(m), state->kernelCutoff);
            }
        }
    }

    return density;
}

float SPHDensitySingle(sf::Vector2f position, Particle* particles, int j, float cutoff) {
    float dX = position.x - particles[j].position.x;
    float dY = position.y - particles[j].position.y;

    if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) return 0.0f;

    float distance = std::sqrt(dX*dX + dY*dY);

    return particles[j].mass * SPHKernel(distance, cutoff);
}


void applyPressureForces(ProgramState *state) {
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        calculatePressureForce(i, state);
    }
}


void calculatePressureForce(int particleIndex, ProgramState* state) {
    int center = state->particles[particleIndex].partitionIndex;

    for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
            int vectorIndex = center + k*state->partitionResolutionY + l;
            if (vectorIndex < 0 || vectorIndex >= state->partitionSize) continue;

            for (int m = 0; m < state->partition[vectorIndex].size(); m++) {
                int j = state->partition[vectorIndex].at(m);
                if (j == particleIndex) continue;
                calculatePressureForceSingle(particleIndex, j, state->particles, state->kernelCutoff);
            }
        }
    }
}

void calculatePressureForceSingle(int particleIndex, int j, Particle* particles, float cutoff) {
    float scalarPart = particles[j].mass / particles[j].density * (particles[particleIndex].pressureOverDSquared + particles[j].pressureOverDSquared);
    sf::Vector2f pressureForce = scalarPart * SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff);
    particles[particleIndex].forces -= pressureForce;
    particles[j].forces += pressureForce;
}


void applyViscosityForces(ProgramState *state) {
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        calculateViscosityForce(i, state);
    }
}



void calculateViscosityForce(int particleIndex, ProgramState* state) {
    int center = state->particles[particleIndex].partitionIndex;

    for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
            int vectorIndex = center + k*state->partitionResolutionY + l;
            if (vectorIndex < 0 || vectorIndex >= state->partitionSize) continue;

            for (int m = 0; m < state->partition[vectorIndex].size(); m++) {
                int j = state->partition[vectorIndex].at(m);
                if (j == particleIndex) continue;
                calculateViscosityForceSingle(particleIndex, j, state->particles, state->kernelCutoff, state->viscosityCoefficient);
            }
        }
    }
}


void calculateViscosityForceSingle(int particleIndex, int j, Particle* particles, float cutoff, float viscosityCoefficient) {
    float dX = particles[particleIndex].position.x - particles[j].position.x;
    float dY = particles[particleIndex].position.y - particles[j].position.y;

    if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) return;

    float distance = std::sqrt(dX*dX + dY*dY);
    if (distance == 0.0f) return;

    sf::Vector2f viscosityForce = viscosityCoefficient * particles[j].mass * SPHKernelLaplacian(distance, cutoff) / particles[j].density * (particles[j].velocity - particles[particleIndex].velocity);

    // sf::Vector2f viscosityForce = viscosityCoefficient * particles[j].mass / particles[j].density
    //     * (particles[particleIndex].velocity / (particles[particleIndex].density * particles[particleIndex].density)
    //         + particles[j].velocity / (particles[j].density * particles[j].density));

    // sf::Vector2f viscosityForce = - viscosityCoefficient / particles[particleIndex].density
    // * dotProduct(particles[particleIndex].velocity - particles[j].velocity, particles[particleIndex].position - particles[j].position)
    // / (distance * distance) * SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff);

    // sf::Vector2f viscosityForce = -particles[j].mass * viscosityCoefficient
    //     / (particles[particleIndex].density * particles[j].density)
    //     * dotProduct(particles[particleIndex].position - particles[j].position, SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff))
    //     / (distance * distance)
    //     * (particles[particleIndex].velocity - particles[j].velocity);

    particles[particleIndex].forces += viscosityForce;
    particles[j].forces -= viscosityForce;
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

float dotProduct(sf::Vector2f v1, sf::Vector2f v2) {
    return v1.x * v2.x + v1.y * v2.y;
}