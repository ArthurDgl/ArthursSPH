#include <iostream>
#include <SFML/Graphics.hpp>
#include <../../../../libs/imgui/imgui.h>
#include <../../../../libs/imgui-sfml/imgui-SFML.h>
#include <cmath>
#include <random>
#include <omp.h>


struct Particle {
    float mass;
    sf::Vector2f position;
    sf::Vector2f velocity;

    sf::Vector2f predictedPosition;

    sf::Vector2f forces;

    int radius;

    float density;
    float pressure;
    float pressureOverDSquared;

    int partitionIndex;

    bool solid;
};

struct ProgramState {
    unsigned int width;
    unsigned int height;

    sf::RenderWindow* window;

    sf::Clock* deltaClock;
    float deltaTime;

    int particleAmount;
    Particle* particles;

    float predictionAmount;

    float kernelCutoff;

    float pressureCoefficient;
    float pressureExponent;
    float restDensity;

    float viscosityCoefficient;

    float gravitationalPull;

    float wallPushCutoff;
    float wallPushStrength;

    float inflowSpeed;

    std::vector<int>* partition;
    int partitionSize;
    int partitionResolutionX;
    int partitionResolutionY;

    float obstacleDensity;
    bool useCeilFloor;

    bool useSphere;
    float sphereX;
    float sphereY;
    float sphereRadius;

    bool useRect;
    float rectX1;
    float rectY1;
    float rectX2;
    float rectY2;

    bool useTri;
    bool triThirdSegment;
    float triX1;
    float triY1;
    float triX2;
    float triY2;
    float triX3;
    float triY3;
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

void createSphereParticles(ProgramState* state, sf::Vector2f position, float radius, float density);
void createLineParticles(ProgramState* state, sf::Vector2f pointA, sf::Vector2f pointB, float density);
void createRectangleParticles(ProgramState* state, float density);

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
        .width = 1200,
        .height = 600,
        .particleAmount = 3300,
        .predictionAmount = 0.005f,
        .kernelCutoff = 50.0f,
        .pressureCoefficient = 100000.0f,
        .pressureExponent = 5.0f,
        .restDensity = 1.55f,
        .viscosityCoefficient = 400.0f,
        .gravitationalPull = 0.0f,//100.0f,
        .wallPushCutoff = 5.0f,
        .wallPushStrength = 50000.0f,
        .inflowSpeed = 150.0f,
        .obstacleDensity = 0.1f,
        .useCeilFloor = true,
        .useSphere = false,
        .sphereX = 200,
        .sphereY = 300,
        .sphereRadius = 50,
        .useRect = false,
        .rectX1 = 200,
        .rectY1 = 450,
        .rectX2 = 280,
        .rectY2 = 610,
        .useTri = true,
        .triThirdSegment = false,
        .triX1 = 200,
        .triY1 = 600,
        .triX2 = 353,
        .triY2 = 549,
        .triX3 = 425,
        .triY3 = 600,
    };
    state.particles = new Particle[state.particleAmount * 2];

    initializeParticles(&state);

    if (state.useSphere) createSphereParticles(&state, sf::Vector2f(state.sphereX, state.sphereY), state.sphereRadius, state.obstacleDensity);
    if (state.useRect) createRectangleParticles(&state, state.obstacleDensity);
    if (state.useTri) {
        createLineParticles(&state, sf::Vector2f(state.triX1, state.triY1), sf::Vector2f(state.triX2, state.triY2), state.obstacleDensity);
        createLineParticles(&state, sf::Vector2f(state.triX2, state.triY2), sf::Vector2f(state.triX3, state.triY3), state.obstacleDensity);
        if (state.triThirdSegment) createLineParticles(&state, sf::Vector2f(state.triX3, state.triY3), sf::Vector2f(state.triX1, state.triY1), state.obstacleDensity);
    }

    if (state.useCeilFloor) {
        createLineParticles(&state, sf::Vector2f(0, 0), sf::Vector2f(state.width, 0), state.obstacleDensity);
        createLineParticles(&state, sf::Vector2f(0, state.height), sf::Vector2f(state.width, state.height), state.obstacleDensity);
    }

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

        // std::cout << state.particles[0].density << std::endl;
    }

    ImGui::SFML::Shutdown();
    return 0;
}


void render(ProgramState *state) {
    displayGui(state);
    state->window->clear();

    sf::CircleShape circle = sf::CircleShape(state->particles->radius);
    circle.setOrigin(state->particles->radius, state->particles->radius);
    sf::Color color = sf::Color::Black;

    // Render Code here
    for (int i = 0; i < state->particleAmount; i++) {
        circle.setPosition(state->particles[i].position);
        float t = (state->particles[i].velocity.x / (state->inflowSpeed * 1.2f) + 1.0f) / 2.0f;
        if (t < 0) t = 0.0f;
        if (t > 1) t = 1.0f;
        color.r = 255 * t;
        color.g = 255 * (1 - t);
        color.b = 0;

        // if (state->particles[i].velocity.x < 0) {
        //     color.r = 0;
        //     color.g = 255;
        //     color.b = 0;
        // }
        // else {
        //     color.r = 255;
        //     color.g = 0;
        //     color.b = 0;
        // }

        if (state->particles[i].solid) {
            color.r = 0;
            color.g = 0;
            color.b = 255;
        }

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

    ImGui::Begin("SPH Info");
    float fps = std::round(100.0f/deltaTime.asSeconds()) / 100.0f;
    ImGui::Text((std::string("FPS : ") + std::to_string(fps)).c_str());

    // ImGui::NewLine();
    // ImGui::Text("Kernel");
    // ImGui::InputFloat("Cutoff Distance", &state->kernelCutoff);
    //
    // ImGui::NewLine();
    // ImGui::Text("Equation of State");
    // ImGui::InputFloat("Pressure Coefficient", &state->pressureCoefficient);
    // ImGui::InputFloat("Density Exponent", &state->pressureExponent);
    // ImGui::InputFloat("Rest Density", &state->restDensity);
    //
    // ImGui::NewLine();
    // ImGui::Text("Viscosity");
    // ImGui::InputFloat("Viscosity Coefficient", &state->viscosityCoefficient);
    //
    // ImGui::NewLine();
    // ImGui::Text("Gravity");
    // ImGui::InputFloat("Gravitational Pull", &state->gravitationalPull);

    ImGui::End();
}


void initializePartition(ProgramState* state) {
    state->partitionResolutionX = state->width / static_cast<int>(state->kernelCutoff) + 2;
    state->partitionResolutionY = state->height / static_cast<int>(state->kernelCutoff) + 2;

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
        int index = getPartitionIndex(state->particles[i].predictedPosition, state);
        if (index < 0 ||index >= state->partitionSize) {
            std::cout << state->particles[i].forces.x << "," << state->particles[i].forces.y << "\n";

            state->particles[i].position = sf::Vector2f(0, 0);
            state->particles[i].predictedPosition = sf::Vector2f(0, 0);
            state->particles[i].velocity = sf::Vector2f(0,0);
            index = 0;
        }

        state->particles[i].partitionIndex = index;
        state->partition[index].push_back(i);
    }
}


void initializeParticles(ProgramState* state) {
    // Define a no-spawn area for particles, to prevent spawning inside obstacles
    float noX1 = 0;
    float noY1 = 0;
    float noX2 = 0;
    float noY2 = 0;
    float padding = 10;

    if (state->useRect) {
        noX1 = state->rectX1 - padding;
        noY1 = state->rectY1 - padding;
        noX2 = state->rectX2 + padding;
        noY2 = state->rectY2 + padding;
    }

    if (state->useTri) {
        noX1 = std::min(std::min(state->triX1, state->triX2), state->triX3) - padding;
        noY1 = std::min(std::min(state->triY1, state->triY2), state->triY3) - padding;

        noX2 = std::max(std::max(state->triX1, state->triX2), state->triX3) + padding;
        noY2 = std::max(std::max(state->triY1, state->triY2), state->triY3) + padding;
    }

    for (int i = 0; i < state->particleAmount; i++) {
        float rX;
        float rY;
        do {
            rX = randomFloat(5, 1.0f*state->width - 5);
            rY = randomFloat(5, 1.0f*state->height - 5);
        } while (rX > noX1 && rX < noX2 && rY > noY1 && rY < noY2);

        state->particles[i] = Particle{
            .mass = 1.0f,
            .position = sf::Vector2f(rX, rY),
            .velocity = sf::Vector2f(state->inflowSpeed, 0),
            .predictedPosition = sf::Vector2f(rX, rY),
            .forces = sf::Vector2f(0, 0),
            .radius = 3,
            .solid = false
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
        if (state->particles[i].solid) continue;

        state->particles[i].velocity += state->particles[i].forces * (dT / state->particles[i].density);
        state->particles[i].position += state->particles[i].velocity * dT;

        state->particles[i].predictedPosition = state->particles[i].position + state->particles[i].velocity * state->predictionAmount;
    }
}

void applyBoundaryConditions(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        if (state->particles[i].solid) continue;

        // Loopback and Constant Edge Velocity
        if (state->particles[i].position.x > state->width) state->particles[i].position.x -= state->width + state->kernelCutoff/2.0f;
        if (state->particles[i].position.x < 20 || state->particles[i].position.x > state->width - 20) state->particles[i].velocity = {state->inflowSpeed, 0.0f};

        // Constraints in +-Y and -X
        if (state->particles[i].position.y > state->height) state->particles[i].position.y = state->height;
        if (state->particles[i].position.x < -state->kernelCutoff) state->particles[i].position.x = -state->kernelCutoff;
        if (state->particles[i].position.y < 0) state->particles[i].position.y = 0.0f;

        // Constraint on Predicted position, to avoid out of bounds for the partition
        if (state->particles[i].predictedPosition.x > state->width + state->kernelCutoff) state->particles[i].predictedPosition.x = state->width + state->kernelCutoff;
        if (state->particles[i].predictedPosition.x < -state->kernelCutoff) state->particles[i].predictedPosition.x = -state->kernelCutoff;

        // Constraint around spherical obstacle
        if (state->useSphere) {
            float sphereDist = sqrt(pow(state->particles[i].position.x - state->sphereX, 2.0) + pow(state->particles[i].position.y - state->sphereY, 2.0));
            float error = state->particles[i].radius + state->sphereRadius - sphereDist;
            if (error > 0) {
                state->particles[i].position += (state->particles[i].position - sf::Vector2f(state->sphereX, state->sphereY)) * (error / sphereDist);
            }
        }
        if (state->useRect) {
            float padding = 5;
            if (state->particles[i].position.x >= state->rectX1 - padding && state->particles[i].position.x <= state->rectX2 + padding
                && state->particles[i].position.y >= state->rectY1 - padding && state->particles[i].position.y <= state->rectY2 + padding) {
                if (state->particles[i].position.x - state->rectX1 > state->rectX2 - state->particles[i].position.x) {
                    state->particles[i].position.x = state->rectX2 + padding*2;
                }
                else {
                    state->particles[i].position.x = state->rectX1 - padding*2;
                }
            }
        }
    }
}

void applyWallForces(ProgramState* state) {
    for (int i = 0; i < state->particleAmount; i++) {
        if (state->particles[i].solid) continue;

        sf::Vector2f wallForce = sf::Vector2f(0, 0);
        //if (state->particles[i].position.x < 0 + state->wallPushCutoff) wallForce.x += state->wallPushStrength;
        if (state->particles[i].position.y < state->wallPushCutoff) wallForce.y += state->wallPushStrength;
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

// void createSphereParticles(ProgramState* state, sf::Vector2f position, float radius, float density) {
//     int radiiAmount = radius * density;
//     for (int i = 0; i < radiiAmount; i++) {
//         float innerRadius = i / density;
//
//         int angleSteps = 6.283185f * innerRadius * density;
//         for (int i = 0; i < angleSteps; i++) {
//             float angle = i / (density * innerRadius);
//             sf::Vector2f relativePosition = innerRadius * sf::Vector2f(cos(angle), sin(angle));
//
//             state->particles[state->particleAmount] = Particle{
//                 .mass = 1.0f,
//                 .position = position + relativePosition,
//                 .velocity = sf::Vector2f(0.0f, 0.0f),
//                 .predictedPosition = position + relativePosition,
//                 .forces = sf::Vector2f(0.0f, 0.0f),
//                 .radius = 2,
//                 .solid = true
//             };
//             state->particleAmount++;
//         }
//     }
// }


void createSphereParticles(ProgramState* state, sf::Vector2f position, float radius, float density) {
    int angleSteps = 6.283185f * radius * density;
    for (int i = 0; i < angleSteps; i++) {
        float angle = i / (density * radius);
        sf::Vector2f relativePosition = radius * sf::Vector2f(cos(angle), sin(angle));

        state->particles[state->particleAmount] = Particle{
            .mass = 1.0f,
            .position = position + relativePosition,
            .velocity = sf::Vector2f(0.0f, 0.0f),
            .predictedPosition = position + relativePosition,
            .forces = sf::Vector2f(0.0f, 0.0f),
            .radius = 2,
            .solid = true
        };
        state->particleAmount++;
    }
}


void createLineParticles(ProgramState* state, sf::Vector2f pointA, sf::Vector2f pointB, float density) {
    float distance = sqrt(pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2));
    float steps = distance * density;
    for (int i = 0; i < steps; i++) {
        float t = i / steps;
        sf::Vector2f pos = pointA * (1 - t) + pointB * t;

        state->particles[state->particleAmount] = Particle{
            .mass = 1.0f,
            .position = pos,
            .velocity = sf::Vector2f(0.0f, 0.0f),
            .predictedPosition = pos,
            .forces = sf::Vector2f(0.0f, 0.0f),
            .radius = 2,
            .solid = true
        };

        state->particleAmount++;
    }
}


void createRectangleParticles(ProgramState* state, float density) {
    createLineParticles(state, sf::Vector2f(state->rectX1, state->rectY1), sf::Vector2f(state->rectX2, state->rectY1), density);
    createLineParticles(state, sf::Vector2f(state->rectX1, state->rectY1), sf::Vector2f(state->rectX1, state->rectY2), density);

    createLineParticles(state, sf::Vector2f(state->rectX2, state->rectY1), sf::Vector2f(state->rectX2, state->rectY2), density);
    createLineParticles(state, sf::Vector2f(state->rectX1, state->rectY2), sf::Vector2f(state->rectX2, state->rectY2), density);
}


void computeDensities(ProgramState* state) {
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].density = SPHDensity(state->particles[i].predictedPosition, state);
    }
}


void computePressures(ProgramState *state) {
    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < state->particleAmount; i++) {
        state->particles[i].pressure = EOSPressure(state->particles[i].density, state->restDensity, state->pressureCoefficient, state->pressureExponent);
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
    float dX = position.x - particles[j].predictedPosition.x;
    float dY = position.y - particles[j].predictedPosition.y;

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
    sf::Vector2f pressureForce = scalarPart * SPHKernelGradient(particles[particleIndex].predictedPosition, particles[j].predictedPosition, cutoff);
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
    float dX = particles[particleIndex].predictedPosition.x - particles[j].predictedPosition.x;
    float dY = particles[particleIndex].predictedPosition.y - particles[j].predictedPosition.y;

    if (std::abs(dX) >= cutoff || std::abs(dY) >= cutoff) return;

    float distance = std::sqrt(dX*dX + dY*dY);
    if (distance == 0.0f) return;

    // sf::Vector2f viscosityForce = viscosityCoefficient * particles[j].mass * SPHKernelLaplacian(distance, cutoff) / particles[j].density * (particles[j].velocity - particles[particleIndex].velocity);

    // sf::Vector2f viscosityForce = viscosityCoefficient * particles[j].mass / particles[j].density
    //     * (particles[particleIndex].velocity / (particles[particleIndex].density * particles[particleIndex].density)
    //         + particles[j].velocity / (particles[j].density * particles[j].density));

    // sf::Vector2f viscosityForce = - viscosityCoefficient / particles[particleIndex].density
    // * dotProduct(particles[particleIndex].velocity - particles[j].velocity, particles[particleIndex].position - particles[j].position)
    // / (distance * distance) * SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff);

    sf::Vector2f viscosityForce = -particles[j].mass * viscosityCoefficient
        / (particles[particleIndex].density * particles[j].density)
        * dotProduct(particles[particleIndex].position - particles[j].position, SPHKernelGradient(particles[particleIndex].position, particles[j].position, cutoff))
        / (distance * distance + 0.0001f)
        * (particles[particleIndex].velocity - particles[j].velocity);

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