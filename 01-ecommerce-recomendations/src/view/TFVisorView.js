import { View } from './View.js';

export class TFVisorView extends View {
    #weights = null;
    #catalog = [];
    #users = [];
    #logs = [];
    #lossPoints = [];
    #accPoints = [];
    constructor() {
        super();

        tfvis.visor().open();
    }

    renderData(data) {

        this.#weights = data.weights;
        this.#catalog = data.catalog;
        this.#users = data.users;
    }
    resetDashboard() {
        this.#weights = null;
        this.#catalog = [];
        this.#users = [];
        this.#logs = [];
        this.#lossPoints = [];
        this.#accPoints = [];
    }

    handleTrainingLog(log) {
        const { epoch, loss, accuracy } = log;
        this.#lossPoints.push({ x: epoch, y: loss });
        this.#accPoints.push({ x: epoch, y: accuracy });
        this.#logs.push(log);

        tfvis.render.linechart(
            {
                name: 'Model Accuracy',
                tab: 'Training',
                style: { display: 'inline-block', width: '49%' }
            },
            { values: this.#accPoints, series: ['accuracy'] },
            {
                xLabel: 'Epoch (Training Cycles)',
                yLabel: 'Accuracy (%)',
                height: 300
            }
        );

        tfvis.render.linechart(
            {
                name: 'Training Error',
                tab: 'Training',
                style: { display: 'inline-block', width: '49%' }
            },
            { values: this.#lossPoints, series: ['errors'] },
            {
                xLabel: 'Epoch (Training Cycles)',
                yLabel: 'Error Value',
                height: 300
            }
        );

    }




}
